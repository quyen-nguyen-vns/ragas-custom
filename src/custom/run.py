import argparse
import json
import os
from typing import List, Optional, cast

from langchain_community.document_loaders import DirectoryLoader
from langchain_core.callbacks import BaseCallbackHandler
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langfuse import observe
from langfuse.langchain import CallbackHandler
from loguru import logger
from pydantic import SecretStr

from ragas.embeddings import LangchainEmbeddingsWrapper
from ragas.llms import LangchainLLMWrapper
from ragas.testset import TestsetGenerator
from ragas.testset.persona import Persona
from ragas.testset.synthesizers import (
    MultiHopSpecificQuerySynthesizer,
    SingleHopSpecificQuerySynthesizer,
)
from ragas.testset.synthesizers.generate import IncrementalSaveConfig
from ragas.testset.synthesizers.multi_hop import MultiHopAbstractQuerySynthesizer
from settings import settings


def create_predefined_personas(
    file_path: str = "cache/persona.json", num_personas: int = 3
) -> List[Persona]:
    """Create pre-defined personas for testset generation."""

    data = json.load(open(file_path))
    personas = []
    for item in data[:num_personas]:
        personas.append(
            Persona(
                name=item["name"],
                role_description=item["role_description"],
            )
        )

    return personas


@observe(name="testset_generation")
async def run_testset_generation(
    input_name: str,
    dataset_name: str,
    kg_name: str,
    testset_size: int,
    llm_name: str,
    embedding_model_name: str,
    languages: Optional[List[str]] = None,
    single_hop_probability: float = 0.7,
    multi_hop_probability: float = 0.3,
    multi_hop_abstract_probability: float = 0,
    persona_file_path: str = "cache/persona.json",
    num_personas: int = 3,
):
    """Main function to generate test dataset."""

    # Set default languages if not provided
    if languages is None:
        languages = ["en"]

    # Setup LangFuse tracing via environment variables
    langfuse_handler = None
    langfuse_client = None
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

            # Set trace metadata and tags using langfuse client
            from langfuse import Langfuse

            langfuse_client = Langfuse()
            langfuse_client.update_current_trace(
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
        (SingleHopSpecificQuerySynthesizer(llm=generator_llm), single_hop_probability),
        (MultiHopSpecificQuerySynthesizer(llm=generator_llm), multi_hop_probability),
        (
            MultiHopAbstractQuerySynthesizer(llm=generator_llm),
            multi_hop_abstract_probability,
        ),
    ]

    # Ensure kg_store_dir exists
    settings.kg_store_dir.mkdir(parents=True, exist_ok=True)

    # Create pre-defined personas
    predefined_personas = create_predefined_personas(
        file_path=persona_file_path, num_personas=num_personas
    )
    logger.info(f"Created {len(predefined_personas)} pre-defined personas:")
    for persona in predefined_personas:
        logger.info(f"  - {persona.name}: {persona.role_description}")

    # Create the test set generator with custom personas
    generator = TestsetGenerator(
        llm=generator_llm,
        embedding_model=generator_embeddings,
        persona_list=predefined_personas,
    )

    # Configure incremental saving - ensure directory exists
    intermediate_dir = settings.intermediate_dir / dataset_name
    intermediate_dir.mkdir(parents=True, exist_ok=True)

    incremental_config = IncrementalSaveConfig(
        enabled=True,
        save_interval=1,  # Save after every sample
        save_scenarios=True,
        save_samples=True,
        save_partial_datasets=True,
        intermediate_dir=intermediate_dir,
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
        dataset = await generator.generate_with_langchain_docs(
            docs,
            testset_size=testset_size,
            query_distribution=distribution,
            kg_name=kg_name,
            incremental_save_config=incremental_config,
            dataset_name=dataset_name,
            num_personas=num_personas,
            callbacks=callbacks,
            raise_exceptions=False,  # Don't raise exceptions, handle them gracefully
            languages=languages,
        )

        # Save and display the generated dataset
        await save_dataset(dataset=dataset, dataset_name=dataset_name)

    except Exception as e:
        logger.error(f"❌ Error during testset generation: {e}")
        # Don't re-raise the exception, let the process continue
        logger.warning("Continuing with partial dataset generation...")
    finally:
        # Always flush LangFuse traces, even if there was an error
        if langfuse_handler:
            logger.info("Flushing LangFuse traces...")
            try:
                langfuse_handler.flush()
                langfuse_client.flush()
                logger.info("✅ LangFuse traces flushed successfully")
            except Exception as flush_error:
                logger.error(f"❌ Error flushing Langfuse traces: {flush_error}")


@observe(name="save_dataset")
async def save_dataset(dataset, dataset_name: str):
    """Save the generated dataset"""
    logger.info("Saving generated dataset...")

    # Create output directory
    output_dir = settings.dataset_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    try:
        # Convert dataset to JSON format
        if hasattr(dataset, "to_pandas"):
            df = dataset.to_pandas()
            data = df.to_dict("records")
            conversion_method = "pandas"
        elif hasattr(dataset, "samples"):
            # Handle Testset object
            data = []
            for sample in dataset.samples:
                if hasattr(sample, "eval_sample"):
                    # Handle TestsetSample
                    eval_sample = sample.eval_sample
                    sample_data = {
                        "question": eval_sample.question,
                        "answer": eval_sample.answer,
                        "contexts": eval_sample.contexts,
                        "ground_truth": eval_sample.ground_truth,
                        "persona_name": getattr(sample, "persona_name", "unknown"),
                        "synthesizer_name": getattr(
                            sample, "synthesizer_name", "unknown"
                        ),
                        "query_style": getattr(sample, "query_style", "unknown"),
                        "query_length": getattr(sample, "query_length", "unknown"),
                    }
                    data.append(sample_data)
                else:
                    # Handle other sample types
                    data.append(sample.__dict__)
            conversion_method = "testset_samples"
        else:
            # If dataset doesn't have to_pandas method, create data manually
            data = []
            for i, item in enumerate(dataset):
                if hasattr(item, "__dict__"):
                    data.append(item.__dict__)
                else:
                    data.append({"index": i, "content": str(item)})
            conversion_method = "manual"

        # Check if we have any data to save
        if not data:
            logger.warning("No data to save - dataset is empty")
            return

        # Save to JSON
        json_path = output_dir / f"{dataset_name}.json"
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

        logger.info(f"💾 Dataset saved to: {json_path}")
        logger.info(f"📊 Saved {len(data)} samples using {conversion_method} method")

    except Exception as e:
        logger.error(f"❌ Error saving dataset: {e}")
        logger.error(f"Dataset type: {type(dataset)}")
        logger.error(
            f"Dataset attributes: {dir(dataset) if hasattr(dataset, '__dict__') else 'No attributes'}"
        )
        raise


if __name__ == "__main__":
    import asyncio

    parser = argparse.ArgumentParser(
        description="Generate test dataset using Ragas testset generator"
    )

    # Required arguments
    parser.add_argument(
        "--input-name",
        type=str,
        required=True,
        help="Name of the input directory containing documents",
    )
    parser.add_argument(
        "--dataset-name",
        type=str,
        required=True,
        help="Name for the generated dataset",
    )
    parser.add_argument(
        "--kg-name",
        type=str,
        required=True,
        help="Name for the knowledge graph",
    )
    parser.add_argument(
        "--testset-size",
        type=int,
        required=True,
        help="Number of test samples to generate",
    )
    parser.add_argument(
        "--llm-name",
        type=str,
        required=True,
        help="Name of the LLM model to use (e.g., 'gemini-2.0-flash')",
    )
    parser.add_argument(
        "--embedding-model-name",
        type=str,
        required=True,
        help="Name of the embedding model to use (e.g., 'models/text-embedding-004')",
    )

    # Optional arguments with defaults
    parser.add_argument(
        "--languages",
        type=str,
        nargs="+",
        default=["en"],
        help="List of languages for generation (default: ['en'])",
    )
    parser.add_argument(
        "--single-hop-probability",
        type=float,
        default=0.7,
        help="Probability for single-hop queries (default: 0.7)",
    )
    parser.add_argument(
        "--multi-hop-probability",
        type=float,
        default=0.3,
        help="Probability for multi-hop specific queries (default: 0.3)",
    )
    parser.add_argument(
        "--multi-hop-abstract-probability",
        type=float,
        default=0.0,
        help="Probability for multi-hop abstract queries (default: 0.0)",
    )
    parser.add_argument(
        "--persona-file-path",
        type=str,
        default="cache/persona.json",
        help="Path to the persona JSON file (default: 'cache/persona.json')",
    )
    parser.add_argument(
        "--num-personas",
        type=int,
        default=3,
        help="Number of personas to use from the persona file (default: 3)",
    )

    args = parser.parse_args()

    asyncio.run(
        run_testset_generation(
            input_name=args.input_name,
            dataset_name=args.dataset_name,
            kg_name=args.kg_name,
            testset_size=args.testset_size,
            llm_name=args.llm_name,
            embedding_model_name=args.embedding_model_name,
            languages=args.languages,
            persona_file_path=args.persona_file_path,
            num_personas=args.num_personas,
            single_hop_probability=args.single_hop_probability,
            multi_hop_probability=args.multi_hop_probability,
            multi_hop_abstract_probability=args.multi_hop_abstract_probability,
        )
    )
