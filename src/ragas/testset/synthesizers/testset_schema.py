from __future__ import annotations

import typing as t
from dataclasses import dataclass, field
from datetime import datetime
from uuid import uuid4

from pydantic import BaseModel, Field

from ragas.cost import CostCallbackHandler, TokenUsage
from ragas.dataset_schema import (
    BaseSample,
    EvaluationDataset,
    MultiTurnSample,
    RagasDataset,
    SingleTurnSample,
)


class TestsetSample(BaseSample):
    """
    Represents a sample in a test set.

    Attributes
    ----------
    eval_sample : Union[SingleTurnSample, MultiTurnSample]
        The evaluation sample, which can be either a single-turn or multi-turn sample.
    synthesizer_name : str
        The name of the synthesizer used to generate this sample.
    persona_name : str
        The name of the persona used to generate this sample.
    query_style : str
        The style of the query used to generate this sample.
    query_length : str
        The length of the query used to generate this sample.
    source_node_ids : list
        List of IDs of the source nodes used to generate this sample.
    source_node_types : list
        List of types of the source nodes (document or chunk).
    source_document_metadata : list
        List of metadata from the original documents (file path, source, etc.).
    source_content_preview : list
        List of previews of the source content used (first 200 characters each).
    """

    eval_sample: t.Union[SingleTurnSample, MultiTurnSample]
    synthesizer_name: str
    persona_name: str = ""
    query_style: str = ""
    query_length: str = ""
    source_node_ids: t.List[str] = Field(default_factory=list)
    source_node_types: t.List[str] = Field(default_factory=list)
    source_document_metadata: t.List[dict] = Field(default_factory=list)
    source_content_preview: t.List[str] = Field(default_factory=list)


class TestsetPacket(BaseModel):
    """
    A packet of testset samples to be uploaded to the server.
    """

    samples_original: t.List[TestsetSample]
    run_id: str
    created_at: str = Field(default_factory=lambda: datetime.now().isoformat())


@dataclass
class Testset(RagasDataset[TestsetSample]):
    """
    Represents a test set containing multiple test samples.

    Attributes
    ----------
    samples : List[TestsetSample]
        A list of TestsetSample objects representing the samples in the test set.
    """

    samples: t.List[TestsetSample]
    run_id: str = field(default_factory=lambda: str(uuid4()), repr=False, compare=False)
    cost_cb: t.Optional[CostCallbackHandler] = field(default=None, repr=False)

    def to_evaluation_dataset(self) -> EvaluationDataset:
        """
        Converts the Testset to an EvaluationDataset.
        """
        return EvaluationDataset(
            samples=[sample.eval_sample for sample in self.samples]
        )

    def to_list(self) -> t.List[t.Dict]:
        """
        Converts the Testset to a list of dictionaries.
        """
        list_dict = []
        for sample in self.samples:
            sample_dict = sample.eval_sample.model_dump(exclude_none=True)
            sample_dict["synthesizer_name"] = sample.synthesizer_name
            sample_dict["persona_name"] = sample.persona_name
            sample_dict["query_style"] = sample.query_style
            sample_dict["query_length"] = sample.query_length
            sample_dict["source_node_ids"] = sample.source_node_ids
            sample_dict["source_node_types"] = sample.source_node_types
            sample_dict["source_document_metadata"] = sample.source_document_metadata
            sample_dict["source_content_preview"] = sample.source_content_preview
            list_dict.append(sample_dict)
        return list_dict

    @classmethod
    def from_list(cls, data: t.List[t.Dict]) -> Testset:
        """
        Converts a list of dictionaries to a Testset.
        """
        # first create the samples
        samples = []
        for sample in data:
            synthesizer_name = sample["synthesizer_name"]
            persona_name = sample.get("persona_name", "")
            query_style = sample.get("query_style", "")
            query_length = sample.get("query_length", "")
            source_node_ids = sample.get("source_node_ids", [])
            source_node_types = sample.get("source_node_types", [])
            source_document_metadata = sample.get("source_document_metadata", [])
            source_content_preview = sample.get("source_content_preview", [])
            # remove the synthesizer name, persona name, query_style, query_length, and source fields from the sample
            sample.pop("synthesizer_name")
            sample.pop("persona_name", None)
            sample.pop("query_style", None)
            sample.pop("query_length", None)
            sample.pop("source_node_ids", None)
            sample.pop("source_node_types", None)
            sample.pop("source_document_metadata", None)
            sample.pop("source_content_preview", None)
            # the remaining sample is the eval_sample
            eval_sample = sample

            # if user_input is a list it is MultiTurnSample
            if "user_input" in eval_sample and not isinstance(
                eval_sample.get("user_input"), list
            ):
                eval_sample = SingleTurnSample(**eval_sample)
            else:
                eval_sample = MultiTurnSample(**eval_sample)

            samples.append(
                TestsetSample(
                    eval_sample=eval_sample,
                    synthesizer_name=synthesizer_name,
                    persona_name=persona_name,
                    query_style=query_style,
                    query_length=query_length,
                    source_node_ids=source_node_ids,
                    source_node_types=source_node_types,
                    source_document_metadata=source_document_metadata,
                    source_content_preview=source_content_preview,
                )
            )
        # then create the testset
        return Testset(samples=samples)

    def total_tokens(self) -> t.Union[t.List[TokenUsage], TokenUsage]:
        """
        Compute the total tokens used in the evaluation.
        """
        if self.cost_cb is None:
            raise ValueError(
                "The Testset was not configured for computing cost. Please provide a token_usage_parser function to TestsetGenerator to compute cost."
            )
        return self.cost_cb.total_tokens()

    def total_cost(
        self,
        cost_per_input_token: t.Optional[float] = None,
        cost_per_output_token: t.Optional[float] = None,
    ) -> float:
        """
        Compute the total cost of the evaluation.
        """
        if self.cost_cb is None:
            raise ValueError(
                "The Testset was not configured for computing cost. Please provide a token_usage_parser function to TestsetGenerator to compute cost."
            )
        return self.cost_cb.total_cost(
            cost_per_input_token=cost_per_input_token,
            cost_per_output_token=cost_per_output_token,
        )

    @classmethod
    def from_annotated(cls, path: str) -> Testset:
        """
        Loads a testset from an annotated JSON file.
        """
        import json

        with open(path, "r") as f:
            annotated_testset = json.load(f)

        samples = []
        for sample in annotated_testset:
            if sample["approval_status"] == "approved":
                samples.append(TestsetSample(**sample))
        return cls(samples=samples)
