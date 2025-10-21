import hashlib
import json
import os
from pathlib import Path
from typing import Dict, List, Literal, Optional

import asyncio
from loguru import logger
from pydantic import BaseModel, Field
from pydantic import SecretStr
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage
from langchain_core.output_parsers import PydanticOutputParser

from settings import settings


class QualityEvaluation(BaseModel):
    """Structured output model for LLM quality evaluation."""
    
    quality: Literal["pass", "out-of-domain", "irrelevant-answer", "low-quality-answer"] = Field(
        description="Quality rating for the question-answer pair"
    )
    reasoning: str = Field(
        description="Detailed reasoning for why this quality rating was assigned"
    )


class DatasetQualityEvaluator:
    """Evaluates dataset quality using LLM with caching mechanism."""
    
    def __init__(self, cache_dir: str = "cache/tagging_cache"):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize LLM
        if not settings.gemini_api_key:
            raise ValueError("GEMINI_API_KEY environment variable is required")
        
        self.llm = ChatGoogleGenerativeAI(
            model="gemini-2.0-flash",
            google_api_key=SecretStr(settings.gemini_api_key),
            temperature=0.1,  # Low temperature for consistent evaluation
        )
        
        # Setup structured output parser
        self.parser = PydanticOutputParser(pydantic_object=QualityEvaluation)
        
        # Statistics tracking
        self.stats = {
            "total": 0,
            "cached": 0,
            "new": 0,
            "pass": 0,
            "out-of-domain": 0,
            "irrelevant-answer": 0,
            "low-quality-answer": 0,
        }
    
    def _generate_cache_key(self, user_input: str, reference: str) -> str:
        """Generate unique cache key from user_input and reference."""
        content = f"{user_input}|||{reference}"
        return hashlib.md5(content.encode()).hexdigest()
    
    def _get_cache_path(self, cache_key: str) -> Path:
        """Get cache file path for a given cache key."""
        return self.cache_dir / f"{cache_key}.json"
    
    def _load_from_cache(self, cache_key: str) -> Optional[QualityEvaluation]:
        """Load evaluation result from cache if exists."""
        cache_path = self._get_cache_path(cache_key)
        if cache_path.exists():
            try:
                with open(cache_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                return QualityEvaluation(**data)
            except Exception as e:
                logger.warning(f"Failed to load cache for {cache_key}: {e}")
        return None
    
    def _save_to_cache(self, cache_key: str, evaluation: QualityEvaluation):
        """Save evaluation result to cache."""
        cache_path = self._get_cache_path(cache_key)
        try:
            with open(cache_path, 'w', encoding='utf-8') as f:
                json.dump(evaluation.model_dump(), f, ensure_ascii=False, indent=2)
        except Exception as e:
            logger.error(f"Failed to save cache for {cache_key}: {e}")
    
    def _create_evaluation_prompt(self, user_input: str, reference: str) -> str:
        """Create structured prompt for quality evaluation."""
        return f"""
You are an expert evaluator for agriculture pest and disease datasets. Your task is to evaluate the quality of question-answer pairs.

EVALUATION CRITERIA:
1. "pass": Question is about agriculture pest/disease AND answer correctly addresses the question
2. "out-of-domain": Question is NOT about agriculture pest/disease domain
3. "irrelevant-answer": Question is on-topic but answer doesn't address it
4. "low-quality-answer": Answer is partially correct but incomplete or unclear

QUESTION: {user_input}

ANSWER: {reference}

Please evaluate this question-answer pair and provide your assessment in the following JSON format:
{{
    "quality": "one of: pass, out-of-domain, irrelevant-answer, low-quality-answer",
    "reasoning": "detailed explanation of your evaluation"
}}

Focus on:
- Whether the question relates to agriculture pest and disease management
- Whether the answer properly addresses the specific question asked
- The completeness and accuracy of the answer
"""
    
    async def evaluate_single(self, user_input: str, reference: str) -> QualityEvaluation:
        """Evaluate a single question-answer pair."""
        cache_key = self._generate_cache_key(user_input, reference)
        
        # Check cache first
        cached_result = self._load_from_cache(cache_key)
        if cached_result:
            self.stats["cached"] += 1
            logger.info(f"Using cached result for question: {user_input[:50]}...")
            return cached_result
        
        # Generate new evaluation
        self.stats["new"] += 1
        logger.info(f"Evaluating new question: {user_input[:50]}...")
        
        try:
            prompt = self._create_evaluation_prompt(user_input, reference)
            message = HumanMessage(content=prompt)
            
            # Get LLM response
            response = await self.llm.ainvoke([message])
            
            # Parse structured output
            content = response.content if isinstance(response.content, str) else str(response.content)
            evaluation = self.parser.parse(content)
            
            # Save to cache
            self._save_to_cache(cache_key, evaluation)
            
            return evaluation
            
        except Exception as e:
            logger.error(f"Error evaluating question: {e}")
            # Return default evaluation on error
            return QualityEvaluation(
                quality="low-quality-answer",
                reasoning=f"Error during evaluation: {str(e)}"
            )
    
    async def evaluate_dataset(
        self, 
        input_path: str = "cache/eval_set_v0_with_sources.json",
        output_path: str = "cache/eval_set_v0_with_quality.json"
    ):
        """Evaluate entire dataset with caching and resume capability."""
        
        # Load input dataset
        logger.info(f"Loading dataset from: {input_path}")
        with open(input_path, 'r', encoding='utf-8') as f:
            dataset = json.load(f)
        
        self.stats["total"] = len(dataset)
        logger.info(f"Total questions to evaluate: {self.stats['total']}")
        
        # Process each data point
        for i, data_point in enumerate(dataset):
            logger.info(f"Processing {i+1}/{self.stats['total']}")
            
            user_input = data_point.get("user_input", "")
            reference = data_point.get("reference", "")
            
            if not user_input or not reference:
                logger.warning(f"Skipping data point {i+1}: missing user_input or reference")
                data_point["quality"] = "low-quality-answer"
                data_point["quality_reasoning"] = "Missing user_input or reference"
                continue
            
            # Evaluate quality
            evaluation = await self.evaluate_single(user_input, reference)
            
            # Add quality fields to data point
            data_point["quality"] = evaluation.quality
            data_point["quality_reasoning"] = evaluation.reasoning
            
            # Update statistics
            self.stats[evaluation.quality] += 1
            
            # Log progress
            if (i + 1) % 10 == 0:
                logger.info(f"Progress: {i+1}/{self.stats['total']} | "
                          f"Cached: {self.stats['cached']} | "
                          f"New: {self.stats['new']}")
        
        # Save output dataset
        logger.info(f"Saving evaluated dataset to: {output_path}")
        output_path_obj = Path(output_path)
        output_path_obj.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(dataset, f, ensure_ascii=False, indent=2)
        
        # Print final statistics
        self._print_statistics()
    
    def _print_statistics(self):
        """Print evaluation statistics."""
        logger.info("=" * 60)
        logger.info("EVALUATION STATISTICS")
        logger.info("=" * 60)
        logger.info(f"Total questions: {self.stats['total']}")
        logger.info(f"Cached evaluations: {self.stats['cached']}")
        logger.info(f"New evaluations: {self.stats['new']}")
        logger.info("")
        logger.info("Quality Distribution:")
        logger.info(f"  pass: {self.stats['pass']} ({self.stats['pass']/self.stats['total']*100:.1f}%)")
        logger.info(f"  out-of-domain: {self.stats['out-of-domain']} ({self.stats['out-of-domain']/self.stats['total']*100:.1f}%)")
        logger.info(f"  irrelevant-answer: {self.stats['irrelevant-answer']} ({self.stats['irrelevant-answer']/self.stats['total']*100:.1f}%)")
        logger.info(f"  low-quality-answer: {self.stats['low-quality-answer']} ({self.stats['low-quality-answer']/self.stats['total']*100:.1f}%)")
        logger.info("=" * 60)


async def main():
    """Main execution function."""
    evaluator = DatasetQualityEvaluator()
    
    await evaluator.evaluate_dataset(
        input_path="cache/eval_set_v0_with_sources.json",
        output_path="cache/eval_set_v0_with_quality.json"
    )


if __name__ == "__main__":
    asyncio.run(main())
