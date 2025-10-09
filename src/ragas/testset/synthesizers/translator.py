"""
Language translation module for multi-language testset generation.
Provides LLM-based translation capabilities for questions and answers.
"""

import typing as t
from dataclasses import dataclass
from functools import lru_cache

from ragas.llms import BaseRagasLLM
from ragas.prompt import Prompt


@dataclass
class LanguageTranslator:
    """
    Translator class that uses LLM to translate text between languages.
    """

    llm: BaseRagasLLM
    source_language: str = "English"
    target_language: str = "Thai"

    def __post_init__(self):
        """Initialize the translation prompt."""
        self.translation_prompt = Prompt(
            instruction=(
                f"Translate the following text from {self.source_language} to {self.target_language}. "
                f"Maintain the original meaning, tone, and context. "
                f"Return only the translated text without any additional explanations or formatting."
            )
        )

    async def translate(self, text: str) -> str:
        """
        Translate a single text string.

        Args:
            text: The text to translate

        Returns:
            The translated text
        """
        if not text or not text.strip():
            return text

        try:
            # Format the prompt with the text to translate
            formatted_prompt = (
                self.translation_prompt.instruction + f"\n\nText to translate:\n{text}"
            )

            # Create a PromptValue object for the LLM
            from langchain_core.prompt_values import StringPromptValue

            prompt_value = StringPromptValue(text=formatted_prompt)

            # Generate translation using the LLM's agenerate_text method
            response = await self.llm.agenerate_text(prompt_value)
            translated_text = response.generations[0][0].text.strip()

            return translated_text
        except Exception as e:
            # If translation fails, return original text with a warning
            print(f"Warning: Translation failed for text '{text[:50]}...': {e}")
            return text

    async def translate_list(self, texts: t.List[str]) -> t.List[str]:
        """
        Translate a list of text strings.

        Args:
            texts: List of texts to translate

        Returns:
            List of translated texts
        """
        if not texts:
            return texts

        translated_texts = []
        for text in texts:
            translated = await self.translate(text)
            translated_texts.append(translated)

        return translated_texts

    @lru_cache(maxsize=1000)
    def _cached_translate(self, text: str) -> str:
        """
        Cached version of translate for synchronous use.
        Note: This is a placeholder - actual caching would need async support.
        """
        return text


class MultiLanguageTranslator:
    """
    Manages translation to multiple target languages.
    """

    def __init__(self, llm: BaseRagasLLM, source_language: str = "English"):
        self.llm = llm
        self.source_language = source_language
        self._translators: t.Dict[str, LanguageTranslator] = {}

    def get_translator(self, target_language: str) -> LanguageTranslator:
        """
        Get or create a translator for the target language.

        Args:
            target_language: The target language code (e.g., 'th', 'vi')

        Returns:
            LanguageTranslator instance for the target language
        """
        if target_language not in self._translators:
            # Map language codes to full names
            language_names = {
                "th": "Thai",
                "vi": "Vietnamese",
                "ja": "Japanese",
                "ko": "Korean",
                "zh": "Chinese",
                "es": "Spanish",
                "fr": "French",
                "de": "German",
                "it": "Italian",
                "pt": "Portuguese",
                "ru": "Russian",
                "ar": "Arabic",
                "hi": "Hindi",
            }

            target_language_name = language_names.get(
                target_language.lower(), target_language.title()
            )

            self._translators[target_language] = LanguageTranslator(
                llm=self.llm,
                source_language=self.source_language,
                target_language=target_language_name,
            )

        return self._translators[target_language]

    async def translate_to_languages(
        self, text: str, target_languages: t.List[str]
    ) -> t.Dict[str, str]:
        """
        Translate text to multiple target languages.

        Args:
            text: The text to translate
            target_languages: List of target language codes

        Returns:
            Dictionary mapping language codes to translated text
        """
        if not text or not text.strip():
            return {lang: text for lang in target_languages}

        translations = {}
        for lang in target_languages:
            if lang == "en":  # Skip English as it's the source
                translations[lang] = text
                continue

            translator = self.get_translator(lang)
            translated = await translator.translate(text)
            translations[lang] = translated

        return translations

    async def translate_list_to_languages(
        self, texts: t.List[str], target_languages: t.List[str]
    ) -> t.Dict[str, t.List[str]]:
        """
        Translate a list of texts to multiple target languages.

        Args:
            texts: List of texts to translate
            target_languages: List of target language codes

        Returns:
            Dictionary mapping language codes to lists of translated texts
        """
        if not texts:
            return {lang: texts for lang in target_languages}

        translations = {}
        for lang in target_languages:
            if lang == "en":  # Skip English as it's the source
                translations[lang] = texts
                continue

            translator = self.get_translator(lang)
            translated = await translator.translate_list(texts)
            translations[lang] = translated

        return translations
