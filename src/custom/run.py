import json
import os

from langchain_community.document_loaders import DirectoryLoader
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from loguru import logger
from pydantic import SecretStr

from ragas.embeddings import LangchainEmbeddingsWrapper
from ragas.llms import LangchainLLMWrapper
from ragas.testset import TestsetGenerator
from ragas.testset.synthesizers import (
    MultiHopAbstractQuerySynthesizer,
    MultiHopSpecificQuerySynthesizer,
    SingleHopSpecificQuerySynthesizer,
)
from ragas.testset.synthesizers.generate import IncrementalSaveConfig
from src.settings import settings


async def run_testset_generation(
    input_name: str,
    dataset_name: str,
    kg_name: str,
    testset_size: int,
    llm_name: str,
    embedding_model_name: str,
):
    """Main function to generate test dataset."""
    # Setup LangSmith tracing if configured
    if settings.langsmith_api_key and settings.langsmith_tracing.lower() == "true":
        os.environ["LANGCHAIN_TRACING_V2"] = "true"
        os.environ["LANGCHAIN_ENDPOINT"] = "https://api.smith.langchain.com"
        os.environ["LANGCHAIN_API_KEY"] = settings.langsmith_api_key
        os.environ["LANGCHAIN_PROJECT"] = settings.langsmith_project
        logger.info(
            f"LangSmith tracing enabled for project: {settings.langsmith_project}"
        )
    else:
        logger.info("LangSmith tracing disabled")

    # Validate API key
    if not settings.gemini_api_key:
        logger.error(
            "Gemini API key not found. Please set GEMINI_API_KEY environment variable."
        )
        return

    # Load documents
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

    # Create the test set generator with custom personas
    generator = TestsetGenerator(
        llm=generator_llm,
        embedding_model=generator_embeddings,
        persona_list=None,
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

    dataset = generator.generate_with_langchain_docs(
        docs,
        testset_size=testset_size,
        query_distribution=distribution,
        kg_name=kg_name,
        incremental_save_config=incremental_config,
        dataset_name=dataset_name,
        num_personas=3,
    )
    # Log the type of dataset returned
    logger.info(f"Dataset type: {type(dataset)}")
    logger.info(f"Dataset attributes: {dir(dataset)}")

    # Save and display the generated dataset
    await save_dataset(dataset=dataset, dataset_name=dataset_name)


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
    else:
        # If dataset doesn't have to_pandas method, create data manually
        data = []
        for i, item in enumerate(dataset):
            if hasattr(item, "__dict__"):
                data.append(item.__dict__)
            else:
                data.append({"index": i, "content": str(item)})

    # Save to JSON
    json_path = output_dir / f"{dataset_name}.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    logger.info(f"💾 Dataset saved to: {json_path}")


if __name__ == "__main__":
    import asyncio

    input_name = "input_17_pest_and_disease"
    dataset_name = "pad-17-0-20"
    kg_name = "pad-17-0"
    testset_size = 20
    llm_name = "gemini-2.5-flash"
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
