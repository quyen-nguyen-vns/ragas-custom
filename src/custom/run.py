import json
import os
from typing import List, Optional, cast

from langchain_community.document_loaders import DirectoryLoader
from langchain_core.callbacks import BaseCallbackHandler
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langfuse.callback import CallbackHandler
from langfuse.decorators import langfuse_context, observe
from loguru import logger
from pydantic import SecretStr

from ragas.embeddings import LangchainEmbeddingsWrapper
from ragas.llms import LangchainLLMWrapper
from ragas.testset import TestsetGenerator
from ragas.testset.persona import Persona
from ragas.testset.synthesizers import (
    MultiHopAbstractQuerySynthesizer,
    MultiHopSpecificQuerySynthesizer,
    SingleHopSpecificQuerySynthesizer,
)
from ragas.testset.synthesizers.generate import IncrementalSaveConfig
from src.settings import settings


def create_predefined_personas() -> List[Persona]:
    """Create pre-defined personas for testset generation."""

    personas = [
        Persona(
            name="Newbie Farmer",
            role_description="A beginner farmer with limited experience in agriculture, just starting their farming journey. Has basic knowledge of plant care but lacks experience in identifying pests and diseases. Eager to learn and asks many questions about farming practices. Needs simple, clear explanations and step-by-step guidance. Often confused by technical terms and prefers practical, easy-to-understand advice. Limited budget and resources, wants to learn cost-effective solutions. Relies heavily on advice from experienced farmers and agricultural resources.",
        ),
        Persona(
            name="Experienced Farmer",
            role_description="A seasoned farmer with 15+ years of experience in crop cultivation, pest management, and sustainable farming practices. Has worked with various crops including rice, vegetables, and fruits. Experienced in identifying plant diseases, managing pests naturally, and optimizing crop yields through traditional and modern techniques. Practical and experience-based thinking, prefers proven methods but open to new techniques that show clear benefits. Focuses on cost-effective solutions and long-term soil health. Limited budget for expensive treatments, prefers environmentally-friendly solutions, needs methods that work in local climate conditions.",
        ),
        Persona(
            name="Durian Orchard Manager",
            role_description="A specialized orchard manager focused on durian cultivation, with expertise in tropical fruit tree management and commercial durian production. Manages a 50-hectare durian orchard with multiple varieties. Expert in durian tree care, flowering cycles, fruit development, and post-harvest handling. Knowledgeable about durian-specific pests and diseases. Detail-oriented and systematic approach, focuses on maximizing fruit quality and yield. Stays updated with latest research on durian cultivation and market trends. Must maintain high fruit quality standards for premium market, seasonal labor management challenges, weather-dependent operations, need to balance organic practices with commercial viability.",
        ),
    ]

    return personas


@observe(name="testset_generation")
async def run_testset_generation(
    input_name: str,
    dataset_name: str,
    kg_name: str,
    testset_size: int,
    llm_name: str,
    embedding_model_name: str,
):
    """Main function to generate test dataset."""

    # Setup LangFuse tracing via environment variables
    langfuse_handler = None
    if settings.langfuse_tracing.lower() == "true":
        if not settings.langfuse_secret_key or not settings.langfuse_public_key:
            logger.error(
                "LangFuse tracing enabled but API keys not found. "
                "Please set LANGFUSE_SECRET_KEY and LANGFUSE_PUBLIC_KEY environment variables."
            )
        else:
            # Set environment variables for LangFuse
            os.environ["LANGFUSE_SECRET_KEY"] = settings.langfuse_secret_key
            os.environ["LANGFUSE_PUBLIC_KEY"] = settings.langfuse_public_key
            os.environ["LANGFUSE_HOST"] = settings.langfuse_host
            logger.info(f"LangFuse tracing enabled for dataset: {dataset_name}")

            # Set trace metadata and tags using langfuse_context
            langfuse_context.update_current_trace(
                name=f"testset_generation_{dataset_name}",
                metadata={
                    "dataset_name": dataset_name,
                    "input_name": input_name,
                    "testset_size": testset_size,
                    "llm_name": llm_name,
                    "embedding_model": embedding_model_name,
                    "kg_name": kg_name,
                },
                tags=["testset-generation", "ragas", dataset_name],
            )

            # Initialize Langfuse CallbackHandler for LangChain tracing
            # This will automatically trace all LangChain LLM calls
            langfuse_handler = CallbackHandler()
            logger.info("✅ Langfuse CallbackHandler initialized successfully")
    else:
        logger.info("LangFuse tracing disabled")

    # Validate API key
    if not settings.gemini_api_key:
        logger.error(
            "Gemini API key not found. Please set GEMINI_API_KEY environment variable."
        )
        return

    # Load documents with tracing
    path = settings.data_input_dir / input_name
    loader = DirectoryLoader(str(path), glob="*.md")
    docs = loader.load()

    if not docs:
        logger.warning(f"No documents found in {path}")
        return

    logger.info(f"Loaded {len(docs)} documents")

    generator_llm = LangchainLLMWrapper(
        ChatGoogleGenerativeAI(
            model=llm_name,
            google_api_key=SecretStr(settings.gemini_api_key),  # type: ignore
        )
    )

    generator_embeddings = LangchainEmbeddingsWrapper(
        GoogleGenerativeAIEmbeddings(
            model=embedding_model_name,
            google_api_key=SecretStr(settings.gemini_api_key),
        )
    )

    # # define distribution of queries
    distribution = [
        (SingleHopSpecificQuerySynthesizer(llm=generator_llm), 0.7),
        (MultiHopSpecificQuerySynthesizer(llm=generator_llm), 0.2),
        (MultiHopAbstractQuerySynthesizer(llm=generator_llm), 0.1),
    ]

    # Create pre-defined personas
    predefined_personas = create_predefined_personas()
    logger.info(f"Created {len(predefined_personas)} pre-defined personas:")
    for persona in predefined_personas:
        logger.info(f"  - {persona.name}: {persona.role_description}")

    # Create the test set generator with custom personas
    generator = TestsetGenerator(
        llm=generator_llm,
        embedding_model=generator_embeddings,
        persona_list=predefined_personas,
    )

    # Configure incremental saving
    incremental_config = IncrementalSaveConfig(
        enabled=True,
        save_interval=1,  # Save after every sample
        save_scenarios=True,
        save_samples=True,
        save_partial_datasets=True,
        intermediate_dir=settings.intermediate_dir / dataset_name,
        cleanup_on_completion=False,  # Keep intermediate files for inspection
    )

    logger.info("Starting dataset generation with incremental saving...")
    logger.info(f"Intermediate results will be saved to: {settings.intermediate_dir}")

    # Prepare callbacks for Ragas - cast to BaseCallbackHandler for type compatibility
    callbacks: Optional[List[BaseCallbackHandler]] = (
        [cast(BaseCallbackHandler, langfuse_handler)] if langfuse_handler else None
    )

    try:
        # Generate dataset with tracing
        dataset = generator.generate_with_langchain_docs(
            docs,
            testset_size=testset_size,
            query_distribution=distribution,
            kg_name=kg_name,
            incremental_save_config=incremental_config,
            dataset_name=dataset_name,
            num_personas=3,
            callbacks=callbacks,
        )
        # Log the type of dataset returned
        logger.info(f"Dataset type: {type(dataset)}")
        logger.info(f"Dataset attributes: {dir(dataset)}")

        # Save and display the generated dataset
        await save_dataset(dataset=dataset, dataset_name=dataset_name)

    except Exception as e:
        logger.error(f"❌ Error during testset generation: {e}")
        raise  # Re-raise the exception after logging
    finally:
        # Always flush LangFuse traces, even if there was an error
        if langfuse_handler:
            logger.info("Flushing LangFuse traces...")
            try:
                langfuse_handler.flush()
                langfuse_context.flush()
                logger.info("✅ LangFuse traces flushed successfully")
            except Exception as flush_error:
                logger.error(f"❌ Error flushing Langfuse traces: {flush_error}")


@observe(name="save_dataset")
async def save_dataset(dataset, dataset_name: str):
    """Save the generated dataset"""
    logger.info("Saving generated dataset...")

    # Create output directory
    output_dir = settings.dataset_dir
    output_dir.mkdir(exist_ok=True)

    # Convert dataset to JSON format
    if hasattr(dataset, "to_pandas"):
        df = dataset.to_pandas()
        data = df.to_dict("records")
        conversion_method = "pandas"
    else:
        # If dataset doesn't have to_pandas method, create data manually
        data = []
        for i, item in enumerate(dataset):
            if hasattr(item, "__dict__"):
                data.append(item.__dict__)
            else:
                data.append({"index": i, "content": str(item)})
        conversion_method = "manual"

    # Save to JSON
    json_path = output_dir / f"{dataset_name}.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

    logger.info(f"💾 Dataset saved to: {json_path}")
    logger.info(f"📊 Saved {len(data)} samples using {conversion_method} method")


if __name__ == "__main__":
    import asyncio

    input_name = "input_17_pest_and_disease"
    kg_name = "pad_17doc_dedup"
    testset_size = 100
    dataset_name = f"pad_17doc_{testset_size}_1"
    llm_name = "gemini-2.0-flash"
    embedding_model_name = "models/text-embedding-004"
    asyncio.run(
        run_testset_generation(
            input_name=input_name,
            dataset_name=dataset_name,
            kg_name=kg_name,
            testset_size=testset_size,
            llm_name=llm_name,
            embedding_model_name=embedding_model_name,
        )
    )
