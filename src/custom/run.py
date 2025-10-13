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
            name="Experienced Farmer",
            role_description="A seasoned farmer with 15+ years of experience in crop cultivation, pest management, and sustainable farming practices. Has worked with various crops including rice, vegetables, and fruits. Experienced in identifying plant diseases, managing pests naturally, and optimizing crop yields through traditional and modern techniques. Practical and experience-based thinking, prefers proven methods but open to new techniques that show clear benefits. Focuses on cost-effective solutions and long-term soil health. Limited budget for expensive treatments, prefers environmentally-friendly solutions, needs methods that work in local climate conditions.",
        ),
        Persona(
            name="Durian Orchard Manager",
            role_description="A specialized orchard manager focused on durian cultivation, with expertise in tropical fruit tree management and commercial durian production. Manages a 50-hectare durian orchard with multiple varieties. Expert in durian tree care, flowering cycles, fruit development, and post-harvest handling. Knowledgeable about durian-specific pests and diseases. Detail-oriented and systematic approach, focuses on maximizing fruit quality and yield. Stays updated with latest research on durian cultivation and market trends. Must maintain high fruit quality standards for premium market, seasonal labor management challenges, weather-dependent operations, need to balance organic practices with commercial viability.",
        ),
        Persona(
            name="Urban Home Gardener",
            role_description="An individual living in the city with limited space for farming, mainly growing vegetables, herbs, and small fruit trees in pots or rooftop gardens. Has moderate knowledge of plant care but struggles with pest control in confined environments. Interested in organic methods and DIY solutions due to concerns about chemical use in small spaces. Wants practical, space-saving techniques and cost-effective tools. Values quick tips and step-by-step visual instructions. Often shares experiences online and learns from gardening communities and social media.",
        ),
        Persona(
            name="Agri-Entrepreneur",
            role_description="A young business-minded individual aiming to build a profitable farming venture. Focused on high-value crops such as organic vegetables, specialty fruits, or greenhouse farming. Very interested in using technology, IoT devices, and data to optimize crop production. Has limited hands-on farming experience but strong knowledge of market trends and customer needs. Prefers structured advice and proven business models over trial-and-error farming. Concerned about investment risk, return on investment, and scaling the business. Seeks innovative but practical farming strategies.",
        ),
        Persona(
            name="Village Elder Farmer",
            role_description="An older farmer with decades of experience in traditional farming methods. Deeply knowledgeable about local soil, seasonal cycles, and indigenous farming practices. Prefers tried-and-true methods but curious about how modern solutions might complement traditional practices. Often respected as a mentor in the community, sharing wisdom with younger farmers. Has limited access to modern equipment and digital platforms. Needs explanations that bridge traditional knowledge with new approaches. Values community well-being and sustainable practices over purely commercial gains.",
        ),
        Persona(
            name="Smallholder Durian Grower",
            role_description="A farmer who owns a small durian plot of 2–5 hectares, growing mixed varieties. Has basic knowledge of durian cultivation but struggles with pest and disease management. Concerned about cost of fertilizers and pesticides, prefers low-cost, practical solutions. Often learns from neighbors or local cooperatives. Needs simple, actionable guidance on pruning, fertilization schedules, and pest prevention to improve fruit quality and yield while keeping expenses low.",
        ),
        Persona(
            name="Commercial Durian Exporter",
            role_description="A business-oriented farmer managing durian orchards for export markets such as China and Singapore. Focused on premium durian varieties like Musang King or Monthong. Knowledgeable about durian flowering cycles, fruit grading, and post-harvest handling standards required for export. Interested in advanced farming practices such as irrigation systems, nutrient monitoring, and integrated pest management. Needs advice on meeting international quality certifications, controlling pesticide residues, and ensuring consistent fruit supply.",
        ),
        Persona(
            name="Durian Research Specialist",
            role_description="An agricultural researcher working closely with universities or government institutes, specializing in durian breeding, disease resistance, and soil management. Familiar with latest scientific findings on Phytophthora, stem canker, and other durian-specific diseases. Advocates for sustainable and environmentally friendly farming practices. Needs detailed technical data, scientific references, and innovative methods to share with farmers. Plays a key role in bridging research with practical orchard management.",
        ),
        Persona(
            name="Durian Enthusiast Hobbyist",
            role_description="An urban professional who loves durians and has recently started planting a few durian trees in their backyard or small family plot. Limited farming experience, relies heavily on online tutorials and community forums. Needs beginner-friendly, step-by-step advice on soil preparation, watering schedules, and early-stage pest control. Focused on personal satisfaction and enjoying homegrown durians rather than commercial profit. Curious about using modern tools like gardening apps or smart irrigation kits.",
        ),
        Persona(
            name="Durian Cooperative Leader",
            role_description="A mid-scale farmer managing 10-20 hectares and actively involved in a local durian growers' cooperative. Experienced in orchard management and familiar with common pests and diseases. Works to unify local farmers for better bargaining power, shared resources, and collective solutions to pest outbreaks. Needs practical, scalable strategies that can be communicated to multiple farmers. Strong interest in government policies, subsidies, and group certifications for sustainable durian farming.",
        ),
        Persona(
            name="Government Extension Officer",
            role_description="An agricultural officer responsible for advising durian farmers in a rural province. Has good general knowledge of horticulture and some training in durian-specific cultivation practices. Provides workshops, distributes manuals, and connects farmers to research institutions. Needs concise, science-backed information that can be easily translated into farmer-friendly language. Balances between promoting modern techniques and respecting local farming traditions. Focused on improving community livelihoods and increasing durian yields for regional markets.",
        ),
        Persona(
            name="Newbie Farmer",
            role_description="A beginner farmer with limited experience in agriculture, just starting their farming journey. Has basic knowledge of plant care but lacks experience in identifying pests and diseases. Eager to learn and asks many questions about farming practices. Needs simple, clear explanations and step-by-step guidance. Often confused by technical terms and prefers practical, easy-to-understand advice. Limited budget and resources, wants to learn cost-effective solutions. Relies heavily on advice from experienced farmers and agricultural resources.",
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
    languages: Optional[List[str]] = None,
):
    """Main function to generate test dataset."""

    # Set default languages if not provided
    if languages is None:
        languages = ["en"]

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
        dataset = await generator.generate_with_langchain_docs(
            docs,
            testset_size=testset_size,
            query_distribution=distribution,
            kg_name=kg_name,
            incremental_save_config=incremental_config,
            dataset_name=dataset_name,
            num_personas=3,
            callbacks=callbacks,
            raise_exceptions=False,  # Don't raise exceptions, handle them gracefully
            languages=languages,
        )
        # Log the type of dataset returned
        logger.info(f"Dataset type: {type(dataset)}")
        logger.info(f"Dataset attributes: {dir(dataset)}")

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

    input_name = "input_17_pest_and_disease"
    kg_name = "pad_17doc_dedup"
    testset_size = 100
    index = 5
    dataset_name = f"pad_17doc_{testset_size}_{index}"
    llm_name = "gemini-2.0-flash"
    embedding_model_name = "models/text-embedding-004"
    # Configure languages - add "th" for Thai translations
    languages = ["en", "th"]  # English and Thai

    asyncio.run(
        run_testset_generation(
            input_name=input_name,
            dataset_name=dataset_name,
            kg_name=kg_name,
            testset_size=testset_size,
            llm_name=llm_name,
            embedding_model_name=embedding_model_name,
            languages=languages,
        )
    )
