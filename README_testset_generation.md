# Testset Generation Guide

This guide explains how to use the testset generation functionality in `run.py` to automatically generate evaluation datasets from your documents.

## Overview

The testset generation tool uses Ragas to create high-quality test datasets from your documents. It supports:

- **Multi-language generation**: Generate queries in multiple languages (e.g., English and Thai)
- **Custom personas**: Use predefined personas to generate diverse, realistic queries
- **Query types**: Single-hop, multi-hop specific, and multi-hop abstract queries
- **Incremental saving**: Save progress incrementally to avoid data loss
- **Knowledge graph caching**: Cached knowledge graphs for faster subsequent runs
- **LangFuse integration**: Optional observability and tracing support

## Prerequisites

1. **Python Environment**: Python 3.11+ with `uv` package manager
2. **API Keys**: 
   - `GEMINI_API_KEY`: Required for Google Gemini LLM and embeddings
   - `LANGFUSE_SECRET_KEY` and `LANGFUSE_PUBLIC_KEY` (optional): For LangFuse tracing
3. **Input Documents**: Markdown files (`.md`) in the input directory
4. **Persona File**: JSON file with persona definitions (default: `cache/persona.json`)

## Setup

### 1. Environment Variables

Set the required environment variables:

```bash
export GEMINI_API_KEY="your-gemini-api-key"

# Optional: For LangFuse tracing
export LANGFUSE_SECRET_KEY="your-secret-key"
export LANGFUSE_PUBLIC_KEY="your-public-key"
export LANGFUSE_HOST="https://cloud.langfuse.com"  # or your self-hosted instance
export LANGFUSE_TRACING="true"  # Set to "true" to enable tracing
```

### 2. Directory Structure

The tool expects the following directory structure:

```
cache/
├── data/
│   ├── input/           # Input documents go here
│   │   └── PL-DR-DP-BK-01/
│   │       └── *.md
│   ├── dataset/         # Generated datasets saved here
│   ├── intermediate/    # Intermediate results during generation
│   └── kg/              # Knowledge graph cache
└── persona.json         # Persona definitions
```

Directories are automatically created if they don't exist.

### 3. Persona File

Create a `persona.json` file with persona definitions. Example:

```json
[
    {
        "name": "Experienced Farmer",
        "role_description": "A seasoned farmer with 15+ years of experience..."
    },
    {
        "name": "Durian Orchard Manager",
        "role_description": "A specialized orchard manager focused on durian cultivation..."
    }
]
```

See `cache/persona.json` for a complete example with 12 predefined personas.

## Usage

### Command Line Interface

Run the script directly with Python:

```bash
export PYTHONPATH=$PWD
uv run python -m src.custom.run \
  --input-name PL-DR-DP-BK-01 \
  --dataset-name PL-DR-DP-BK-01_10_0 \
  --kg-name PL-DR-DP-BK-01 \
  --testset-size 10 \
  --llm-name "gemini-2.0-flash" \
  --embedding-model-name "models/text-embedding-004" \
  --languages en th \
  --num-personas 3 \
  --persona-file-path "cache/persona.json" \
  --single-hop-probability 0.7 \
  --multi-hop-probability 0.3 \
  --multi-hop-abstract-probability 0
```

### Using the Shell Script

Alternatively, use the provided shell script:

```bash
chmod +x src/custom/run.sh
./src/custom/run.sh
```

Edit `src/custom/run.sh` to customize the parameters.

## Parameters

### Required Parameters

| Parameter | Description | Example |
|-----------|-------------|---------|
| `--input-name` | Name of the input directory containing documents | `PL-DR-DP-BK-01` |
| `--dataset-name` | Name for the generated dataset | `PL-DR-DP-BK-01_10_0` |
| `--kg-name` | Name for the knowledge graph (used for caching) | `PL-DR-DP-BK-01` |
| `--testset-size` | Number of test samples to generate | `10` |
| `--llm-name` | LLM model name | `gemini-2.0-flash` |
| `--embedding-model-name` | Embedding model name | `models/text-embedding-004` |

### Optional Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--languages` | `["en"]` | List of languages for generation (space-separated) |
| `--single-hop-probability` | `0.7` | Probability for single-hop queries (0.0-1.0) |
| `--multi-hop-probability` | `0.3` | Probability for multi-hop specific queries (0.0-1.0) |
| `--multi-hop-abstract-probability` | `0.0` | Probability for multi-hop abstract queries (0.0-1.0) |
| `--persona-file-path` | `cache/persona.json` | Path to persona JSON file |
| `--num-personas` | `3` | Number of personas to use from the persona file |

**Note**: The probabilities should sum to approximately 1.0 for proper distribution.

## Query Types

### Single-Hop Queries
Simple, direct questions that can be answered from a single document chunk.

**Example**: "What are the symptoms of durian root rot?"

**Use Case**: Testing basic retrieval and answer generation capabilities.

### Multi-Hop Specific Queries
Complex questions requiring information from multiple related document chunks.

**Example**: "Compare the treatment methods for durian root rot and stem canker."

**Use Case**: Testing multi-step reasoning and information synthesis.

### Multi-Hop Abstract Queries
Abstract questions requiring reasoning across multiple concepts.

**Example**: "How do environmental factors affect durian disease management strategies?"

**Use Case**: Testing advanced reasoning and cross-domain knowledge integration.

## Output

### Generated Dataset

The dataset is saved as JSON in `cache/data/dataset/{dataset_name}.json`:

```json
[
    {
        "question": "What are the symptoms of durian root rot?",
        "answer": "...",
        "contexts": ["...", "..."],
        "ground_truth": "...",
        "persona_name": "Experienced Farmer",
        "synthesizer_name": "SingleHopSpecificQuerySynthesizer",
        "query_style": "specific",
        "query_length": "medium"
    }
]
```

### Intermediate Results

During generation, intermediate results are saved to:
- `cache/data/intermediate/{dataset_name}/`

This includes:
- Scenarios
- Individual samples
- Partial datasets

Intermediate files are kept for inspection (not cleaned up automatically).

### Knowledge Graph

The knowledge graph is cached to:
- `cache/data/kg/{kg_name}.json`

If the knowledge graph already exists, it will be reused, significantly speeding up subsequent runs with the same input documents.

### Neo4j
You can use script `src/custom/import_to_neo4j.py` to import KG into Neo4j for visualization.

## Workflow

1. **Document Loading**: Loads markdown files from `cache/data/input/{input_name}/`
2. **Knowledge Graph Creation**: 
   - If cached KG exists, loads it
   - Otherwise, creates KG from documents and applies transforms
   - Saves KG for future use
3. **Persona Loading**: Loads personas from JSON file
4. **Testset Generation**: Generates queries using configured synthesizers
5. **Incremental Saving**: Saves progress after each sample
6. **Dataset Export**: Converts and saves final dataset as JSON

## Advanced Configuration

### Custom Transforms

By default, the tool uses `default_transforms()` which:
- Extracts headlines from documents
- Splits documents into chunks (min: 500 tokens, max: 1000 tokens)
- Extracts summaries, keyphrases, and entities
- Builds similarity relationships

You can customize this by modifying the code to pass custom transforms to `generate_with_langchain_docs()`.

### Chunk Size Configuration

The default chunk size is:
- **Minimum**: 500 tokens
- **Maximum**: 1000 tokens

This is configured in `default_transforms()` in `src/ragas/testset/transforms/default.py`. To customize, create your own transforms with a custom `HeadlineSplitter`.

## Troubleshooting

### Common Issues

1. **No documents found**
   - Ensure documents are in `cache/data/input/{input_name}/`
   - Check that files have `.md` extension
   - Verify directory path is correct

2. **API key errors**
   - Verify `GEMINI_API_KEY` is set correctly
   - Check API key has sufficient quota

3. **Knowledge graph creation fails**
   - Ensure documents are long enough (> 500 tokens for most)
   - Check LLM API is accessible
   - Review logs for specific error messages

4. **Empty dataset**
   - Check if testset generation completed successfully
   - Review intermediate results in `cache/data/intermediate/`
   - Verify personas are correctly formatted

### Logging

The tool uses `loguru` for logging. Logs include:
- Document loading progress
- Persona creation
- Knowledge graph operations
- Testset generation progress
- Error messages

## Examples

## Performance Tips

1. **Reuse Knowledge Graphs**: Use the same `--kg-name` for the same input documents to avoid rebuilding the KG
2. **Incremental Saving**: The tool saves after each sample, so you can resume from partial results
3. **Batch Processing**: For large datasets, generate in smaller batches and combine results
4. **Monitor API Usage**: Large testset sizes can consume significant API quota

## Best Practices

1. **Start Small**: Begin with small testset sizes (10-20) to validate setup
2. **Persona Diversity**: Use multiple personas to generate diverse queries
3. **Query Distribution**: Balance query types based on your use case
4. **Document Quality**: Ensure input documents are well-structured and relevant
5. **Regular Caching**: Reuse knowledge graphs for the same document sets
6. **Incremental Work**: Generate datasets incrementally to avoid losing progress

## Support

For issues or questions:
1. Check the logs for error messages
2. Review intermediate results in `cache/data/intermediate/`
3. Verify all environment variables are set correctly
4. Ensure input documents meet minimum requirements

## Documentation
- [Ragas's Core Concepts](https://docs.ragas.io/en/stable/concepts/test_data_generation/rag/)
- [Confluence](https://vns-site.atlassian.net/wiki/spaces/AIS/pages/74678914/KES+Evaluation+Set+-+V0)
