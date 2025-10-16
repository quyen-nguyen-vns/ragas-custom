import json

from neo4j import GraphDatabase
from tqdm import tqdm


def connect_to_neo4j(
    uri: str = "bolt://localhost:11678",
    username: str = "neo4j",
    password: str = "password",
):
    """Connect to Neo4j database."""
    return GraphDatabase.driver(uri, auth=(username, password))


def get_document_source_for_node(driver, node_id: str) -> str:
    """Query Neo4j to find the document source for a given node ID."""
    with driver.session() as session:
        query = """
        MATCH (n:DOCUMENT)-[]->(m) 
        WHERE m.id = $node_id 
        RETURN n.document_metadata as metadata
        """
        result = session.run(query, node_id=node_id)
        record = result.single()

        if record and record["metadata"]:
            # Extract source from document_metadata
            metadata = record["metadata"]
            source_path = None

            if isinstance(metadata, dict) and "source" in metadata:
                source_path = metadata["source"]
            elif isinstance(metadata, str):
                # If metadata is stored as JSON string, parse it
                try:
                    parsed_metadata = json.loads(metadata)
                    source_path = parsed_metadata.get("source", None)
                except json.JSONDecodeError:
                    return None

            if source_path:
                # Extract filename from path and replace .md with .pdf
                filename = source_path.split("/")[
                    -1
                ]  # Get the last part after splitting by '/'
                if filename.endswith(".md"):
                    filename = filename[:-3] + ".pdf"  # Replace .md with .pdf
                return filename

        return None


def main():
    """Main function to process eval_set_v0.json and find document sources."""
    print("Loading eval_set_v0.json...")
    eval_data = json.load(
        open("cache/data/dataset/eval_set_v0.json", "r", encoding="utf-8")
    )
    print(f"Loaded {len(eval_data)} evaluation data points")

    # Connect to Neo4j
    print("Connecting to Neo4j...")
    driver = connect_to_neo4j()
    print("✓ Connected to Neo4j!")

    # Extract unique source_node_ids
    source_node_ids = set()
    for data_point in eval_data:
        if "source_node_ids" in data_point:
            source_node_ids.update(data_point["source_node_ids"])

    print(f"Found {len(source_node_ids)} unique source node IDs")

    # Create mapping from node_id to document source
    node_id_to_source = {}

    print("Querying Neo4j for document sources...")
    for node_id in tqdm(source_node_ids):
        source = get_document_source_for_node(driver, node_id)
        node_id_to_source[node_id] = source

    # Close Neo4j connection
    driver.close()

    # Create enhanced eval data with document sources
    enhanced_eval_data = []
    for data_point in tqdm(eval_data, desc="Enhancing eval data"):
        enhanced_point = data_point.copy()

        # Add document sources for each source_node_id
        if "source_node_ids" in data_point:
            document_sources = []
            for node_id in data_point["source_node_ids"]:
                source = node_id_to_source.get(node_id, None)
                if source is not None:  # Only add non-None sources
                    document_sources.append(source)
            # Remove duplicates while preserving order
            unique_sources = []
            seen = set()
            for source in document_sources:
                if source not in seen:
                    unique_sources.append(source)
                    seen.add(source)
            enhanced_point["source_document_sources"] = unique_sources

        enhanced_eval_data.append(enhanced_point)

    # Save enhanced eval data
    print("Saving enhanced eval data...")
    with open(
        "cache/data/dataset/eval_set_v0_with_sources.json", "w", encoding="utf-8"
    ) as f:
        json.dump(enhanced_eval_data, f, ensure_ascii=False, indent=2)

    # Save node_id to source mapping
    print("Saving node_id to source mapping...")
    with open("cache/data/kg/node_id_to_source.json", "w", encoding="utf-8") as f:
        json.dump(node_id_to_source, f, ensure_ascii=False, indent=2)

    # Print summary statistics
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"Total evaluation data points: {len(eval_data)}")
    print(f"Unique source node IDs: {len(source_node_ids)}")
    print(
        f"Nodes with document sources found: {sum(1 for source in node_id_to_source.values() if source is not None)}"
    )
    print(
        f"Nodes without document sources: {sum(1 for source in node_id_to_source.values() if source is None)}"
    )

    # Show some examples of document sources
    print("\nSample document sources:")
    count = 0
    for node_id, source in node_id_to_source.items():
        if source and count < 5:
            print(f"  {node_id}: {source}")
            count += 1

    print("\nFiles saved:")
    print("  - cache/data/dataset/eval_set_v0_with_sources.json (enhanced eval data)")
    print("  - cache/data/kg/node_id_to_source.json (node_id to source mapping)")


if __name__ == "__main__":
    main()
