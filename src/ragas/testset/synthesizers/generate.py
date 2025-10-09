from __future__ import annotations

import json
import logging
import random
import typing as t
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from uuid import UUID

from langchain_core.callbacks import BaseCallbackManager
from langchain_core.documents import Document as LCDocument

from ragas._analytics import TestsetGenerationEvent, track
from ragas.callbacks import new_group

logger = logging.getLogger(__name__)
from ragas.cost import TokenUsageParser
from ragas.embeddings.base import (
    BaseRagasEmbeddings,
    LangchainEmbeddingsWrapper,
    LlamaIndexEmbeddingsWrapper,
)
from ragas.executor import Executor
from ragas.llms import BaseRagasLLM, LangchainLLMWrapper, LlamaIndexLLMWrapper
from ragas.run_config import RunConfig
from ragas.testset.graph import KnowledgeGraph, Node, NodeType
from ragas.testset.persona import Persona, generate_personas_from_kg
from ragas.testset.synthesizers import default_query_distribution
from ragas.testset.synthesizers.testset_schema import Testset, TestsetSample
from ragas.testset.synthesizers.translator import MultiLanguageTranslator
from ragas.testset.synthesizers.utils import calculate_split_values
from ragas.testset.transforms import Transforms, default_transforms

if t.TYPE_CHECKING:
    from langchain_core.callbacks import Callbacks
    from langchain_core.embeddings import Embeddings as LangchainEmbeddings
    from langchain_core.language_models import BaseLanguageModel as LangchainLLM
    from llama_index.core.base.embeddings.base import (
        BaseEmbedding as LlamaIndexEmbedding,
    )
    from llama_index.core.base.llms.base import BaseLLM as LlamaIndexLLM
    from llama_index.core.schema import Document as LlamaIndexDocument

    from ragas.embeddings.base import BaseRagasEmbeddings
    from ragas.llms.base import BaseRagasLLM
    from ragas.testset.synthesizers import QueryDistribution
    from ragas.testset.synthesizers.base import BaseScenario


RAGAS_TESTSET_GENERATION_GROUP_NAME = "ragas testset generation"
logger = logging.getLogger(__name__)


class UUIDEncoder(json.JSONEncoder):
    """Custom JSON encoder to handle UUID serialization."""

    def default(self, obj):
        if isinstance(obj, UUID):
            return str(obj)
        return super().default(obj)


@dataclass
class IncrementalSaveConfig:
    """Configuration for incremental saving during testset generation."""

    enabled: bool = True
    save_interval: int = 1  # Save after every N samples
    save_scenarios: bool = True
    save_samples: bool = True
    save_partial_datasets: bool = True
    intermediate_dir: t.Optional[Path] = None
    cleanup_on_completion: bool = False


class IncrementalSaveCallback:
    """Callback system for saving intermediate results during testset generation."""

    def __init__(self, config: IncrementalSaveConfig):
        self.config = config
        self.current_samples = []
        self.current_scenarios = []
        self.sample_count = 0
        self.scenario_count = 0

        # Create intermediate directory structure
        if self.config.enabled and self.config.intermediate_dir:
            self._setup_directories()

    def _setup_directories(self):
        """Create the directory structure for intermediate results."""
        if self.config.intermediate_dir is None:
            return

        base_dir = self.config.intermediate_dir
        base_dir.mkdir(parents=True, exist_ok=True)

        # Create subdirectories for different types of intermediate results
        (base_dir / "scenarios").mkdir(exist_ok=True)
        (base_dir / "samples").mkdir(exist_ok=True)
        (base_dir / "partial_datasets").mkdir(exist_ok=True)
        (base_dir / "knowledge_graphs").mkdir(exist_ok=True)

    def on_scenario_generated(self, scenario, synthesizer_name: str):
        """Called when a scenario is generated."""
        if not self.config.enabled or not self.config.save_scenarios:
            return

        self.current_scenarios.append(
            {
                "scenario": scenario.model_dump()
                if hasattr(scenario, "model_dump")
                else scenario.__dict__,
                "synthesizer_name": synthesizer_name,
                "timestamp": datetime.now().isoformat(),
            }
        )

        self.scenario_count += 1

        # Save scenarios in batches
        if self.scenario_count % 10 == 0:  # Save every 10 scenarios
            self._save_scenarios()

    def on_sample_generated(
        self,
        sample,
        synthesizer_name: str,
        persona_name: str = "",
        query_style: str = "",
        query_length: str = "",
        source_node_ids: t.Optional[t.List[str]] = None,
        source_node_types: t.Optional[t.List[str]] = None,
        source_document_metadata: t.Optional[t.List[dict]] = None,
        source_content_preview: t.Optional[t.List[str]] = None,
    ):
        """Called when a sample (question-answer pair) is generated."""
        if not self.config.enabled or not self.config.save_samples:
            return

        self.current_samples.append(
            {
                "sample": sample.model_dump()
                if hasattr(sample, "model_dump")
                else sample.__dict__,
                "synthesizer_name": synthesizer_name,
                "persona_name": persona_name,
                "query_style": query_style,
                "query_length": query_length,
                "source_node_ids": source_node_ids or [],
                "source_node_types": source_node_types or [],
                "source_document_metadata": source_document_metadata or [],
                "source_content_preview": source_content_preview or [],
                "timestamp": datetime.now().isoformat(),
            }
        )

        self.sample_count += 1

        # Save individual sample
        if self.config.save_samples:
            self._save_individual_sample(
                sample,
                synthesizer_name,
                persona_name,
                query_style,
                query_length,
                source_node_ids,
                source_node_types,
                source_document_metadata,
                source_content_preview,
            )

        # Save partial dataset if interval reached
        if self.sample_count % self.config.save_interval == 0:
            self._save_partial_dataset()

    def on_knowledge_graph_updated(self, kg, step_name: str):
        """Called when knowledge graph is updated at different steps."""
        if not self.config.enabled or self.config.intermediate_dir is None:
            return

        kg_path = (
            self.config.intermediate_dir / "knowledge_graphs" / f"{step_name}_kg.json"
        )
        kg.save(kg_path)
        logger.info(f"Saved knowledge graph at step '{step_name}' to {kg_path}")

    def _save_scenarios(self):
        """Save current scenarios to file."""
        if not self.current_scenarios or self.config.intermediate_dir is None:
            return

        scenarios_path = (
            self.config.intermediate_dir
            / "scenarios"
            / f"scenarios_batch_{self.scenario_count // 10}.json"
        )
        with open(scenarios_path, "w", encoding="utf-8") as f:
            json.dump(
                self.current_scenarios, f, ensure_ascii=False, indent=2, cls=UUIDEncoder
            )

        logger.info(
            f"Saved {len(self.current_scenarios)} scenarios to {scenarios_path}"
        )
        self.current_scenarios.clear()

    def _save_individual_sample(
        self,
        sample,
        synthesizer_name: str,
        persona_name: str = "",
        query_style: str = "",
        query_length: str = "",
        source_node_ids: t.Optional[t.List[str]] = None,
        source_node_types: t.Optional[t.List[str]] = None,
        source_document_metadata: t.Optional[t.List[dict]] = None,
        source_content_preview: t.Optional[t.List[str]] = None,
    ):
        """Save individual sample to file."""
        if self.config.intermediate_dir is None:
            return

        sample_path = (
            self.config.intermediate_dir
            / "samples"
            / f"sample_{self.sample_count:04d}.json"
        )
        sample_data = {
            "sample": sample.model_dump()
            if hasattr(sample, "model_dump")
            else sample.__dict__,
            "synthesizer_name": synthesizer_name,
            "persona_name": persona_name,
            "query_style": query_style,
            "query_length": query_length,
            "source_node_ids": source_node_ids or [],
            "source_node_types": source_node_types or [],
            "source_document_metadata": source_document_metadata or [],
            "source_content_preview": source_content_preview or [],
            "sample_number": self.sample_count,
            "timestamp": datetime.now().isoformat(),
        }

        with open(sample_path, "w", encoding="utf-8") as f:
            json.dump(sample_data, f, ensure_ascii=False, indent=2, cls=UUIDEncoder)

    def _save_partial_dataset(self):
        """Save partial dataset with current samples."""
        if (
            not self.config.save_partial_datasets
            or not self.current_samples
            or self.config.intermediate_dir is None
        ):
            return

        dataset_path = (
            self.config.intermediate_dir
            / "partial_datasets"
            / f"dataset_after_{self.sample_count}_samples.json"
        )

        # Convert samples to the format expected by the final dataset
        dataset_data = []
        for sample_data in self.current_samples:
            dataset_data.append(
                {
                    "user_input": sample_data["sample"].get("user_input", ""),
                    "reference_contexts": sample_data["sample"].get(
                        "reference_contexts", []
                    ),
                    "reference": sample_data["sample"].get("reference", ""),
                    "synthesizer_name": sample_data["synthesizer_name"],
                    "persona_name": sample_data.get("persona_name", ""),
                    "query_style": sample_data.get("query_style", ""),
                    "query_length": sample_data.get("query_length", ""),
                    "source_node_ids": sample_data.get("source_node_ids", []),
                    "source_node_types": sample_data.get("source_node_types", []),
                    "source_document_metadata": sample_data.get(
                        "source_document_metadata", []
                    ),
                    "source_content_preview": sample_data.get(
                        "source_content_preview", []
                    ),
                    "sample_number": sample_data.get("sample_number", 0),
                    "timestamp": sample_data["timestamp"],
                }
            )

        with open(dataset_path, "w", encoding="utf-8") as f:
            json.dump(dataset_data, f, ensure_ascii=False, indent=2, cls=UUIDEncoder)

        logger.info(
            f"Saved partial dataset with {len(dataset_data)} samples to {dataset_path}"
        )

    def finalize(self):
        """Finalize saving and cleanup if configured."""
        # Save any remaining scenarios
        if self.current_scenarios:
            self._save_scenarios()

        # Save final partial dataset
        if self.current_samples:
            self._save_partial_dataset()

        # Cleanup if configured
        if self.config.cleanup_on_completion:
            self._cleanup_intermediate_files()

    def _cleanup_intermediate_files(self):
        """Clean up intermediate files after completion."""
        if self.config.intermediate_dir and self.config.intermediate_dir.exists():
            import shutil

            shutil.rmtree(self.config.intermediate_dir)
            logger.info(
                f"Cleaned up intermediate directory: {self.config.intermediate_dir}"
            )


class IncrementalExecutor:
    """Enhanced Executor that supports incremental saving of results."""

    def __init__(self, base_executor, save_callback: IncrementalSaveCallback):
        self.base_executor = base_executor
        self.save_callback = save_callback
        self._original_submit = base_executor.submit
        self._original_results = base_executor.results

        # Wrap the submit method to track sample generation
        self._wrap_submit_method()

    def _wrap_submit_method(self):
        """Wrap the submit method to add incremental saving."""
        original_submit = self.base_executor.submit

        def wrapped_submit(callable, *args, name=None, **kwargs):
            # Check if this is a sample generation call
            if hasattr(callable, "__name__") and "generate_sample" in str(callable):
                # Extract persona name, style, length, and source info from the scenario argument
                persona_name = ""
                query_style = ""
                query_length = ""
                source_node_ids = []
                source_node_types = []
                source_document_metadata = []
                source_content_preview = []

                if len(args) > 0 and hasattr(args[0], "persona"):
                    persona_name = args[0].persona.name
                if len(args) > 0 and hasattr(args[0], "style"):
                    query_style = args[0].style.value
                if len(args) > 0 and hasattr(args[0], "length"):
                    query_length = args[0].length.value
                if (
                    len(args) > 0
                    and hasattr(args[0], "nodes")
                    and len(args[0].nodes) > 0
                ):
                    # Extract info from all nodes (for both single-hop and multi-hop)
                    for node in args[0].nodes:
                        source_node_ids.append(str(node.id))
                        source_node_types.append(node.type.value)
                        source_metadata = node.properties.get("document_metadata", {})
                        source_document_metadata.append(source_metadata)
                        source_content = node.properties.get("page_content", "")
                        preview = (
                            source_content[:200] + "..."
                            if len(source_content) > 200
                            else source_content
                        )
                        source_content_preview.append(preview)

                # Wrap the callable to save results incrementally
                async def wrapped_callable(*inner_args, **inner_kwargs):
                    result = await callable(*inner_args, **inner_kwargs)
                    # Extract synthesizer name from the callable
                    synthesizer_name = "unknown"
                    if hasattr(callable, "__self__") and hasattr(
                        callable.__self__, "name"
                    ):
                        synthesizer_name = callable.__self__.name
                    self.save_callback.on_sample_generated(
                        result,
                        synthesizer_name,
                        persona_name,
                        query_style,
                        query_length,
                        source_node_ids,
                        source_node_types,
                        source_document_metadata,
                        source_content_preview,
                    )
                    return result

                return original_submit(wrapped_callable, *args, name=name, **kwargs)
            else:
                return original_submit(callable, *args, name=name, **kwargs)

        self.base_executor.submit = wrapped_submit

    def submit(self, callable, *args, name=None, **kwargs):
        """Submit a job with incremental saving support."""
        return self.base_executor.submit(callable, *args, name=name, **kwargs)

    def results(self):
        """Get results and finalize saving."""
        try:
            results = self.base_executor.results()
            return results
        finally:
            # Finalize saving after all results are collected
            self.save_callback.finalize()

    def cancel(self):
        """Cancel execution."""
        return self.base_executor.cancel()

    def is_cancelled(self):
        """Check if cancelled."""
        return self.base_executor.is_cancelled()

    def clear_jobs(self):
        """Clear jobs."""
        return self.base_executor.clear_jobs()

    def __getattr__(self, name):
        """Delegate other attributes to the base executor."""
        return getattr(self.base_executor, name)


@dataclass
class TestsetGenerator:
    """
    Generates an evaluation dataset based on given scenarios and parameters.

    Attributes
    ----------
    llm : BaseRagasLLM
        The language model to use for the generation process.
    knowledge_graph : KnowledgeGraph, default empty
        The knowledge graph to use for the generation process.
    """

    llm: BaseRagasLLM
    embedding_model: BaseRagasEmbeddings
    knowledge_graph: KnowledgeGraph = field(default_factory=KnowledgeGraph)
    persona_list: t.Optional[t.List[Persona]] = None

    @classmethod
    def from_langchain(
        cls,
        llm: LangchainLLM,
        embedding_model: LangchainEmbeddings,
        knowledge_graph: t.Optional[KnowledgeGraph] = None,
    ) -> TestsetGenerator:
        """
        Creates a `TestsetGenerator` from a Langchain LLMs.
        """
        knowledge_graph = knowledge_graph or KnowledgeGraph()
        return cls(
            LangchainLLMWrapper(llm),
            LangchainEmbeddingsWrapper(embedding_model),
            knowledge_graph,
        )

    @classmethod
    def from_llama_index(
        cls,
        llm: LlamaIndexLLM,
        embedding_model: LlamaIndexEmbedding,
        knowledge_graph: t.Optional[KnowledgeGraph] = None,
    ) -> TestsetGenerator:
        """
        Creates a `TestsetGenerator` from a LlamaIndex LLM and embedding model.
        """
        knowledge_graph = knowledge_graph or KnowledgeGraph()
        return cls(
            LlamaIndexLLMWrapper(llm),
            LlamaIndexEmbeddingsWrapper(embedding_model),
            knowledge_graph,
        )

    async def generate_with_langchain_docs(
        self,
        documents: t.Sequence[LCDocument],
        testset_size: int,
        transforms: t.Optional[Transforms] = None,
        transforms_llm: t.Optional[BaseRagasLLM] = None,
        transforms_embedding_model: t.Optional[BaseRagasEmbeddings] = None,
        query_distribution: t.Optional[QueryDistribution] = None,
        run_config: t.Optional[RunConfig] = None,
        callbacks: t.Optional[Callbacks] = None,
        with_debugging_logs=False,
        raise_exceptions: bool = True,
        return_executor: bool = False,
        kg_name: str = "knowledge_graph_test.json",
        dataset_name: str = "data_test",
        incremental_save_config: t.Optional[IncrementalSaveConfig] = None,
        num_personas: int = 3,
        languages: t.Optional[t.List[str]] = None,
    ) -> t.Union[Testset, Executor, IncrementalExecutor]:
        """
        Generates an evaluation dataset based on given Langchain documents and parameters.

        Parameters
        ----------
        documents : Sequence[LCDocument]
            A sequence of Langchain documents to use as source material
        testset_size : int
            The number of test samples to generate
        transforms : Optional[Transforms], optional
            Custom transforms to apply to the documents, by default None
        transforms_llm : Optional[BaseRagasLLM], optional
            LLM to use for transforms if different from instance LLM, by default None
        transforms_embedding_model : Optional[BaseRagasEmbeddings], optional
            Embedding model to use for transforms if different from instance model, by default None
        query_distribution : Optional[QueryDistribution], optional
            Distribution of query types to generate, by default None
        run_config : Optional[RunConfig], optional
            Configuration for the generation run, by default None
        callbacks : Optional[Callbacks], optional
            Callbacks to use during generation, by default None
        with_debugging_logs : bool, optional
            Whether to include debug logs, by default False
        raise_exceptions : bool, optional
            Whether to raise exceptions during generation, by default True
        return_executor : bool, optional
            If True, returns the Executor instance instead of running generation.
            The returned executor can be used to cancel execution by calling executor.cancel().
            To get results, call executor.results(). Default is False.

        Returns
        -------
        Testset or Executor
            If return_executor is False, returns the generated evaluation dataset.
            If return_executor is True, returns the Executor instance for cancellable execution.

        Raises
        ------
        ValueError
            If no LLM or embedding model is provided either during initialization or as arguments
        """

        # force the user to provide an llm and embedding client to prevent use of default LLMs
        if not self.llm and not transforms_llm:
            raise ValueError(
                """An llm client was not provided.
                       Provide an LLM on TestsetGenerator instantiation or as an argument for transforms_llm parameter.
                       Alternatively you can provide your own transforms through the `transforms` parameter."""
            )
        if not self.embedding_model and not transforms_embedding_model:
            raise ValueError(
                """An embedding client was not provided. Provide an embedding through the transforms_embedding_model parameter. Alternatively you can provide your own transforms through the `transforms` parameter."""
            )

        if not transforms:
            transforms = default_transforms(
                documents=list(documents),
                llm=transforms_llm or self.llm,
                embedding_model=transforms_embedding_model or self.embedding_model,
            )

        # convert the documents to Ragas nodes
        nodes = []
        for doc in documents:
            node = Node(
                type=NodeType.DOCUMENT,
                properties={
                    "page_content": doc.page_content,
                    "document_metadata": doc.metadata,
                },
            )
            nodes.append(node)

        # Create knowledge graph with nodes
        kg = KnowledgeGraph(nodes=nodes)

        # Apply transforms to enrich the knowledge graph
        from src.settings import settings

        if kg_name.endswith(".json"):
            knowledge_graph_json_path = settings.kg_store_dir / kg_name
        else:
            knowledge_graph_json_path = settings.kg_store_dir / f"{kg_name}.json"
        if not knowledge_graph_json_path.exists():
            logger.info("Applying transforms to the knowledge graph...")
            from ragas.testset.transforms import apply_transforms

            apply_transforms(kg, transforms, callbacks=callbacks)
            self.knowledge_graph = kg

            # Save the knowledge graph
            logger.info("Saving the knowledge graph...")
            self.knowledge_graph.save(knowledge_graph_json_path)
            logger.info(f"Knowledge graph saved to: {knowledge_graph_json_path}")
            exit()
        else:
            self.knowledge_graph = KnowledgeGraph.load(knowledge_graph_json_path)
            logger.info(f"Knowledge graph loaded from: {knowledge_graph_json_path}")

        logger.info("Knowledge graph created successfully!")

        return await self.generate(
            testset_size=testset_size,
            query_distribution=query_distribution,
            run_config=run_config,
            callbacks=callbacks,
            with_debugging_logs=with_debugging_logs,
            raise_exceptions=raise_exceptions,
            return_executor=return_executor,
            incremental_save_config=incremental_save_config,
            dataset_name=dataset_name,
            num_personas=num_personas,
            languages=languages,
        )

    def generate_with_llamaindex_docs(
        self,
        documents: t.Sequence[LlamaIndexDocument],
        testset_size: int,
        transforms: t.Optional[Transforms] = None,
        transforms_llm: t.Optional[LlamaIndexLLM] = None,
        transforms_embedding_model: t.Optional[LlamaIndexEmbedding] = None,
        query_distribution: t.Optional[QueryDistribution] = None,
        run_config: t.Optional[RunConfig] = None,
        callbacks: t.Optional[Callbacks] = None,
        with_debugging_logs=False,
        raise_exceptions: bool = True,
        dataset_name: str = "data_test",
    ):
        """
        Generates an evaluation dataset based on given scenarios and parameters.
        """

        run_config = run_config or RunConfig()

        # force the user to provide an llm and embedding client to prevent use of default LLMs
        if not self.llm and not transforms_llm:
            raise ValueError(
                "An llm client was not provided. Provide an LLM on TestsetGenerator instantiation or as an argument for transforms_llm parameter. Alternatively you can provide your own transforms through the `transforms` parameter."
            )
        if not self.embedding_model and not transforms_embedding_model:
            raise ValueError(
                "An embedding client was not provided. Provide an embedding through the transforms_embedding_model parameter. Alternatively you can provide your own transforms through the `transforms` parameter."
            )

        if not transforms:
            # use TestsetGenerator's LLM and embedding model if no transforms_llm or transforms_embedding_model is provided
            if transforms_llm is None:
                llm_for_transforms = self.llm
            else:
                llm_for_transforms = LlamaIndexLLMWrapper(transforms_llm)
            if transforms_embedding_model is None:
                embedding_model_for_transforms = self.embedding_model
            else:
                embedding_model_for_transforms = LlamaIndexEmbeddingsWrapper(
                    transforms_embedding_model
                )

            # create the transforms
            transforms = default_transforms(
                documents=[LCDocument(page_content=doc.text) for doc in documents],
                llm=llm_for_transforms,
                embedding_model=embedding_model_for_transforms,
            )

        # convert the documents to Ragas nodes
        nodes = []
        for doc in documents:
            if doc.text is not None and doc.text.strip() != "":
                node = Node(
                    type=NodeType.DOCUMENT,
                    properties={
                        "page_content": doc.text,
                        "document_metadata": doc.metadata,
                    },
                )
                nodes.append(node)

        kg = KnowledgeGraph(nodes=nodes)

        from ragas.testset.transforms import apply_transforms

        # apply transforms and update the knowledge graph
        apply_transforms(kg, transforms, run_config)
        self.knowledge_graph = kg

        return self.generate(
            testset_size=testset_size,
            query_distribution=query_distribution,
            run_config=run_config,
            callbacks=callbacks,
            with_debugging_logs=with_debugging_logs,
            raise_exceptions=raise_exceptions,
            return_executor=False,  # Default value for llamaindex_docs method
            dataset_name=dataset_name,
        )

    async def generate(
        self,
        testset_size: int,
        query_distribution: t.Optional[QueryDistribution] = None,
        num_personas: int = 3,
        run_config: t.Optional[RunConfig] = None,
        batch_size: t.Optional[int] = None,
        callbacks: t.Optional[Callbacks] = None,
        token_usage_parser: t.Optional[TokenUsageParser] = None,
        with_debugging_logs=False,
        raise_exceptions: bool = True,
        return_executor: bool = False,
        incremental_save_config: t.Optional[IncrementalSaveConfig] = None,
        dataset_name: str = "data_test",
        languages: t.Optional[t.List[str]] = None,
    ) -> t.Union[Testset, Executor, IncrementalExecutor]:
        """
        Generate an evaluation dataset based on given scenarios and parameters.

        Parameters
        ----------
        testset_size : int
            The number of samples to generate.
        query_distribution : Optional[QueryDistribution], optional
            A list of tuples containing scenario simulators and their probabilities.
            If None, default simulators will be used.
        num_personas : int, default 3
            The number of personas to generate or use from the persona_list.
        run_config : Optional[RunConfig], optional
            Configuration for running the generation process.
        batch_size: int, optional
            How large should batches be.  If set to None (default), no batching is done.
        callbacks : Optional[Callbacks], optional
            Langchain style callbacks to use for the generation process. You can use
            this to log the generation process or add other metadata.
        token_usage_parser : Optional[TokenUsageParser], optional
            Parse the LLMResult object and return a TokenUsage object. This is used to
            calculate the cost of the generation process.
        with_debugging_logs : bool, default False
            If True, enable debug logging for various components.
        raise_exceptions : bool, default True
            If True, raise exceptions during the generation process.
        return_executor : bool, default False
            If True, returns the Executor instance instead of running generation.
            The returned executor can be used to cancel execution by calling executor.cancel().
            To get results, call executor.results().
        incremental_save_config : Optional[IncrementalSaveConfig], optional
            Configuration for incremental saving of intermediate results during generation.
            If provided, enables step-by-step saving of scenarios, samples, and partial datasets.

        Returns
        -------
        Testset or Executor
            If return_executor is False, returns a dataset containing the generated TestsetSamples.
            If return_executor is True, returns the Executor instance for cancellable execution.

        Notes
        -----
        This function performs the following steps:
        1. Set up scenarios and debug logging if required.
        2. Generate scenarios using an Executor.
        3. Calculate split values for different scenario types.
        4. Generate samples for each scenario.
        5. Compile the results into an EvaluationDataset.

        If incremental_save_config is provided, intermediate results are saved at each step.
        """
        if run_config is not None:
            self.llm.set_run_config(run_config)

        # Setup incremental saving if configured
        save_callback = None
        if incremental_save_config:
            # Set default intermediate directory from settings if not provided
            if incremental_save_config.intermediate_dir is None:
                from src.settings import settings

                incremental_save_config.intermediate_dir = (
                    settings.intermediate_dir / dataset_name
                )

            save_callback = IncrementalSaveCallback(incremental_save_config)
            logger.info(
                f"Incremental saving enabled. Intermediate results will be saved to: {incremental_save_config.intermediate_dir}"
            )

        query_distribution = query_distribution or default_query_distribution(
            self.llm, self.knowledge_graph
        )
        callbacks = callbacks or []

        # dict to store any callbacks we define
        ragas_callbacks = {}
        # set the token usage parser
        if token_usage_parser is not None:
            from ragas.cost import CostCallbackHandler

            cost_cb = CostCallbackHandler(token_usage_parser=token_usage_parser)
            ragas_callbacks["cost_cb"] = cost_cb
        else:
            cost_cb = None

        # append all the ragas_callbacks to the callbacks
        for cb in ragas_callbacks.values():
            if isinstance(callbacks, BaseCallbackManager):
                callbacks.add_handler(cb)
            else:
                callbacks.append(cb)

        # new group for Testset Generation
        testset_generation_rm, testset_generation_grp = new_group(
            name=RAGAS_TESTSET_GENERATION_GROUP_NAME,
            inputs={"testset_size": testset_size},
            callbacks=callbacks,
        )

        if with_debugging_logs:
            # TODO: Edit this before pre-release
            from ragas.utils import patch_logger

            patch_logger("ragas.experimental.testset.synthesizers", logging.DEBUG)
            patch_logger("ragas.experimental.testset.graph", logging.DEBUG)
            patch_logger("ragas.experimental.testset.transforms", logging.DEBUG)

        # Save knowledge graph after persona generation setup
        if save_callback:
            save_callback.on_knowledge_graph_updated(
                self.knowledge_graph, "after_persona_setup"
            )

        if self.persona_list is None:
            self.persona_list = generate_personas_from_kg(
                llm=self.llm,
                kg=self.knowledge_graph,
                num_personas=num_personas,
                callbacks=callbacks,
            )
        else:
            random.shuffle(self.persona_list)

        # ================================================================
        # Save self.persona_list into a JSON file for inspection or reuse
        import json

        from src.settings import settings

        # Use the same intermediate directory as the incremental save config
        if incremental_save_config and incremental_save_config.intermediate_dir:
            persona_json_path = (
                incremental_save_config.intermediate_dir / "persona_list.json"
            )
        else:
            persona_json_path = (
                settings.intermediate_dir / dataset_name / "persona_list.json"
            )

        # Create the directory first
        persona_json_path.parent.mkdir(parents=True, exist_ok=True)

        # Convert persona_list to serializable format (list of dicts)
        persona_data = [p.model_dump() for p in self.persona_list]

        with open(persona_json_path, "w", encoding="utf-8") as f:
            json.dump(persona_data, f, ensure_ascii=False, indent=2, cls=UUIDEncoder)
        # ================================================================

        splits, _ = calculate_split_values(
            [prob for _, prob in query_distribution], testset_size
        )
        # new group for Generation of Scenarios
        scenario_generation_rm, scenario_generation_grp = new_group(
            name="Scenario Generation",
            inputs={"splits": splits},
            callbacks=testset_generation_grp,
        )

        # generate scenarios
        base_exec = Executor(
            desc="Generating Scenarios",
            raise_exceptions=raise_exceptions,
            run_config=run_config,
            keep_progress_bar=False,
            batch_size=batch_size,
        )

        # Wrap with IncrementalExecutor if saving is enabled
        if save_callback:
            exec = IncrementalExecutor(base_exec, save_callback)
        else:
            exec = base_exec
        # generate samples
        splits, _ = calculate_split_values(
            [prob for _, prob in query_distribution], testset_size
        )
        for i, (scenario, _) in enumerate(query_distribution):
            # Wrap scenario generation to save scenarios incrementally
            if save_callback:

                async def wrapped_scenario_generation(
                    synthesizer, n, kg, personas, callbacks
                ):
                    scenarios = await synthesizer._generate_scenarios(
                        n, kg, personas, callbacks
                    )
                    # Save each scenario
                    for scenario_obj in scenarios:
                        save_callback.on_scenario_generated(
                            scenario_obj, synthesizer.name
                        )
                    return scenarios

                exec.submit(
                    wrapped_scenario_generation,
                    scenario,
                    splits[i],
                    self.knowledge_graph,
                    self.persona_list[:num_personas],
                    scenario_generation_grp,
                )
            else:
                exec.submit(
                    scenario._generate_scenarios,
                    splits[i],
                    self.knowledge_graph,
                    self.persona_list[:num_personas],
                    scenario_generation_grp,
                )

        try:
            scenario_sample_list: t.List[t.List[BaseScenario]] = exec.results()
        except Exception as e:
            scenario_generation_rm.on_chain_error(e)
            raise e
        else:
            scenario_generation_rm.on_chain_end(
                outputs={"scenario_sample_list": scenario_sample_list}
            )

        # new group for Generation of Samples
        sample_generation_rm, sample_generation_grp = new_group(
            name="Sample Generation",
            inputs={"scenario_sample_list": scenario_sample_list},
            callbacks=testset_generation_grp,
        )
        base_sample_exec = Executor(
            "Generating Samples",
            raise_exceptions=raise_exceptions,
            run_config=run_config,
            keep_progress_bar=True,
            batch_size=batch_size,
        )

        # Wrap with IncrementalExecutor if saving is enabled
        if save_callback:
            exec = IncrementalExecutor(base_sample_exec, save_callback)
        else:
            exec = base_sample_exec
        additional_testset_info: t.List[t.Dict] = []
        for i, (synthesizer, _) in enumerate(query_distribution):
            for sample in scenario_sample_list[i]:
                exec.submit(
                    synthesizer.generate_sample,
                    scenario=sample,
                    callbacks=sample_generation_grp,
                )
                # Extract source information from all nodes (for both single-hop and multi-hop)
                source_node_ids = []
                source_node_types = []
                source_document_metadata = []
                source_content_preview = []

                for node in sample.nodes:
                    source_node_ids.append(str(node.id))
                    source_node_types.append(node.type.value)
                    source_metadata = node.properties.get("document_metadata", {})
                    source_document_metadata.append(source_metadata)
                    source_content = node.properties.get("page_content", "")
                    preview = (
                        source_content[:200] + "..."
                        if len(source_content) > 200
                        else source_content
                    )
                    source_content_preview.append(preview)

                # fill out the additional info for the TestsetSample
                additional_testset_info.append(
                    {
                        "synthesizer_name": synthesizer.name,
                        "persona_name": sample.persona.name,
                        "query_style": sample.style.value,
                        "query_length": sample.length.value,
                        "source_node_ids": source_node_ids,
                        "source_node_types": source_node_types,
                        "source_document_metadata": source_document_metadata,
                        "source_content_preview": source_content_preview,
                    }
                )

        # Return executor for cancellable execution if requested
        if return_executor:
            return exec

        try:
            eval_samples = exec.results()
        except Exception as e:
            sample_generation_rm.on_chain_error(e)
            raise e
        else:
            sample_generation_rm.on_chain_end(outputs={"eval_samples": eval_samples})

        # build the testset
        testsets = []
        for sample, additional_info in zip(eval_samples, additional_testset_info):
            testsets.append(TestsetSample(eval_sample=sample, **additional_info))

        # Add translations if additional languages are requested
        if languages and len(languages) > 1:
            # Filter out English from target languages
            target_languages = [lang for lang in languages if lang != "en"]
            if target_languages:
                logger.info(f"Adding translations for languages: {target_languages}")

                # Create translator
                translator = MultiLanguageTranslator(
                    self.llm, source_language="English"
                )

                # Translate each sample
                for testset_sample in testsets:
                    eval_sample = testset_sample.eval_sample

                    # Prepare translations for this sample
                    sample_translations = {}

                    # Translate user_input if it exists
                    if hasattr(eval_sample, "user_input") and eval_sample.user_input:
                        user_input_translations = (
                            await translator.translate_to_languages(
                                eval_sample.user_input, target_languages
                            )
                        )
                        for lang in target_languages:
                            if lang not in sample_translations:
                                sample_translations[lang] = {}
                            sample_translations[lang]["user_input"] = (
                                user_input_translations[lang]
                            )

                    # Translate reference if it exists
                    if hasattr(eval_sample, "reference") and eval_sample.reference:
                        reference_translations = (
                            await translator.translate_to_languages(
                                eval_sample.reference, target_languages
                            )
                        )
                        for lang in target_languages:
                            if lang not in sample_translations:
                                sample_translations[lang] = {}
                            sample_translations[lang]["reference"] = (
                                reference_translations[lang]
                            )

                    # Add translation fields to the eval_sample
                    if sample_translations:
                        eval_sample.add_translation_fields(sample_translations)

                logger.info("✅ Translation completed for all samples")

        testset = Testset(samples=testsets, cost_cb=cost_cb)
        testset_generation_rm.on_chain_end({"testset": testset})

        # tracking how many samples were generated
        track(
            TestsetGenerationEvent(
                event_type="testset_generation",
                evolution_names=[
                    e.__class__.__name__.lower() for e, _ in query_distribution
                ],
                evolution_percentages=[p for _, p in query_distribution],
                num_rows=testset_size,
                language="english",
            )
        )
        return testset
