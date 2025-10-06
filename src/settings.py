import enum
from abc import ABC
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


class LogLevel(str, enum.Enum):
    """Possible log levels."""

    NOTSET = "NOTSET"
    DEBUG = "DEBUG"
    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"
    FATAL = "FATAL"


class Environment(str, enum.Enum):
    """Environment types."""

    DEVELOPMENT = "development"
    TESTING = "testing"
    STAGING = "staging"
    PRODUCTION = "production"


class ProjectBaseSettings(BaseSettings, ABC):
    """Base settings for the project."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_prefix="",
        env_file_encoding="utf-8",
        extra="ignore",
    )


class LLMSettings(ProjectBaseSettings):
    """Settings for the LLM-related options."""

    openai_api_key: str = ""
    openai_api_base: str = "https://api.openai.com/v1"
    gemini_api_key: str = ""


class LangSmithSettings(ProjectBaseSettings):
    """Settings for LangSmith observability."""

    langsmith_api_key: str = ""
    langsmith_tracing: str = "false"
    langsmith_project: str = "ragas-testset-generation"


class AWSSettings(ProjectBaseSettings):
    """Settings for the AWS-related options."""

    aws_access_key_id: str = ""
    aws_secret_access_key: str = ""
    aws_session_token: str = ""
    aws_region: str = "us-east-1"
    aws_s3_bucket: str = "document-parser-bucket"
    aws_s3_prefix_parsing: str = "documents-parsing"
    aws_s3_prefix_indexing: str = "documents-indexing"


class DynamoDBSettings(ProjectBaseSettings):
    """Settings for the DynamoDB-related options."""

    dynamodb_table_name_parsing: str = "documents-parsing"
    dynamodb_table_name_indexing: str = "documents-indexing"


class ProjectSettings(LLMSettings, LangSmithSettings, AWSSettings, DynamoDBSettings):
    """Application settings.

    These parameters can be configured
    with environment variables.
    """

    dataset_dir: Path = Path("cache/data/dataset")
    data_input_dir: Path = Path("cache/data/input")
    intermediate_dir: Path = Path("cache/data/intermediate")
    kg_store_dir: Path = Path("cache/data/kg")


settings = ProjectSettings()
