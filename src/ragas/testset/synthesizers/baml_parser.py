"""
BAML-based robust JSON parser for query generation.
This ensures that LLM outputs are correctly parsed even with formatting issues.
"""

import json
import typing as t

from pydantic import BaseModel

from ragas.llms import BaseRagasLLM
from ragas.prompt import PydanticPrompt

# Type variables for generics
InputModel = t.TypeVar("InputModel", bound=BaseModel)
OutputModel = t.TypeVar("OutputModel", bound=BaseModel)


class BAMLRobustParser:
    """
    A robust parser that handles malformed JSON from LLM outputs.
    Uses multiple strategies to extract valid JSON.
    """

    @staticmethod
    def fix_json_string(json_str: str) -> str:
        """Fix common JSON formatting issues in LLM outputs."""
        # Remove markdown code blocks if present
        json_str = json_str.strip()
        if json_str.startswith("```json"):
            json_str = json_str[7:]
        if json_str.startswith("```"):
            json_str = json_str[3:]
        if json_str.endswith("```"):
            json_str = json_str[:-3]
        json_str = json_str.strip()

        # Fix common escape issues
        # Replace unescaped single quotes with escaped ones
        # But be careful not to break already escaped quotes
        lines = json_str.split("\n")
        fixed_lines = []

        for line in lines:
            # Find strings in the line (between double quotes)
            in_string = False
            fixed_line = []
            i = 0
            while i < len(line):
                char = line[i]

                if char == '"' and (i == 0 or line[i - 1] != "\\"):
                    in_string = not in_string
                    fixed_line.append(char)
                elif in_string and char == "\\":
                    # Check if it's a valid escape sequence
                    if i + 1 < len(line):
                        next_char = line[i + 1]
                        # Valid escape sequences: \", \\, \/, \b, \f, \n, \r, \t, \uXXXX
                        if next_char in ['"', "\\", "/", "b", "f", "n", "r", "t", "u"]:
                            fixed_line.append(char)
                        elif next_char == "'":
                            # Invalid escape \' - just use the single quote without backslash
                            i += 1  # Skip the backslash
                            fixed_line.append("'")
                        else:
                            # Invalid escape, escape the backslash
                            fixed_line.append("\\\\")
                    else:
                        # Backslash at end of line, escape it
                        fixed_line.append("\\\\")
                elif in_string and char == "'":
                    # Convert single quotes inside strings to escaped single quotes
                    # In JSON, single quotes don't need escaping, but some LLMs use them incorrectly
                    fixed_line.append(
                        char
                    )  # Keep as is, single quotes are valid in JSON strings
                else:
                    fixed_line.append(char)
                i += 1

            fixed_lines.append("".join(fixed_line))

        return "\n".join(fixed_lines)

    @staticmethod
    def extract_json_object(text: str) -> t.Optional[dict]:
        """Extract JSON object from text, handling various formats."""
        text = text.strip()

        # Try direct parsing first
        try:
            fixed_text = BAMLRobustParser.fix_json_string(text)
            return json.loads(fixed_text)
        except json.JSONDecodeError:
            pass

        # Try to find JSON object in the text
        start_idx = text.find("{")
        end_idx = text.rfind("}")

        if start_idx != -1 and end_idx != -1 and end_idx > start_idx:
            json_str = text[start_idx : end_idx + 1]
            try:
                fixed_json = BAMLRobustParser.fix_json_string(json_str)
                return json.loads(fixed_json)
            except json.JSONDecodeError:
                pass

        # Last resort: try to extract key-value pairs manually
        try:
            return BAMLRobustParser.extract_key_values(text)
        except Exception:
            return None

    @staticmethod
    def extract_key_values(text: str) -> dict:
        """Extract key-value pairs from text manually."""
        result = {}

        # Look for "query": "..." and "answer": "..."
        import re

        # Extract query
        query_match = re.search(
            r'"query"\s*:\s*"([^"]*(?:\\.[^"]*)*)"', text, re.DOTALL
        )
        if query_match:
            raw_query = query_match.group(1)
            # Clean up escape sequences
            raw_query = raw_query.replace('\\"', '"')
            raw_query = raw_query.replace("\\'", "'")  # Fix escaped single quotes
            raw_query = raw_query.replace("\\\\", "\\")
            result["query"] = raw_query

        # Extract answer
        answer_match = re.search(
            r'"answer"\s*:\s*"([^"]*(?:\\.[^"]*)*)"', text, re.DOTALL
        )
        if answer_match:
            raw_answer = answer_match.group(1)
            # Clean up escape sequences
            raw_answer = raw_answer.replace('\\"', '"')
            raw_answer = raw_answer.replace("\\'", "'")  # Fix escaped single quotes
            raw_answer = raw_answer.replace("\\\\", "\\")
            result["answer"] = raw_answer

        if "query" in result and "answer" in result:
            return result

        raise ValueError("Could not extract query and answer from text")


class BAMLEnhancedPrompt(
    PydanticPrompt[InputModel, OutputModel], t.Generic[InputModel, OutputModel]
):
    """
    Enhanced PydanticPrompt that uses BAML-style robust parsing.
    Supports generic type parameters for input and output models.
    """

    async def parse_output_string(  # type: ignore[override]
        self,
        output: str,
        prompt_value: t.Any,
        llm: BaseRagasLLM,
        callbacks: t.Any,
    ) -> BaseModel:
        """Parse output string with robust error handling."""
        # Try the original parser first
        try:
            return await super().parse_output_string(  # type: ignore
                output, prompt_value, llm, callbacks
            )
        except Exception as original_error:
            # If original parsing fails, use robust parser
            try:
                parsed_dict = BAMLRobustParser.extract_json_object(output)
                if parsed_dict:
                    return self.output_model(**parsed_dict)
                else:
                    raise original_error
            except Exception:
                # If robust parser also fails, raise the original error
                raise original_error
