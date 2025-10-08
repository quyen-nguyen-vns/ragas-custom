"""
Import Knowledge Graph to Neo4j

This script imports the knowledge graph from knowledge_graph_test.json into Neo4j.
"""

import json
from typing import Any, Dict

from neo4j import GraphDatabase


class Neo4jImporter:
    def __init__(self, uri: str, username: str, password: str):
        """Initialize Neo4j connection."""
        self.driver = GraphDatabase.driver(uri, auth=(username, password))

    def close(self):
        """Close Neo4j connection."""
        self.driver.close()

    def clear_database(self):
        """Clear all nodes and relationships in the database."""
        with self.driver.session() as session:
            print("Clearing existing data...")
            session.run("MATCH (n) DETACH DELETE n")
            print("Database cleared!")

    def create_constraints(self):
        """Create constraints and indexes for better performance."""
        with self.driver.session() as session:
            print("Creating constraints and indexes...")
            try:
                # Create constraint on node ID
                session.run(
                    "CREATE CONSTRAINT node_id IF NOT EXISTS FOR (n:Node) REQUIRE n.id IS UNIQUE"
                )
                print("  ✓ Created constraint on Node.id")
            except Exception as e:
                print(f"  Note: Constraint already exists or error: {e}")

    def import_nodes(self, nodes: list):
        """Import nodes into Neo4j."""
        with self.driver.session() as session:
            print(f"\nImporting {len(nodes)} nodes...")
            count = 0

            for node in nodes:
                # Convert properties to Neo4j-compatible format
                properties = {}
                for key, value in node.get("properties", {}).items():
                    # Skip very large properties or convert to string
                    if isinstance(value, (list, dict)):
                        # For lists and dicts, store as JSON string if small enough
                        if key == "summary_embedding":
                            # Skip embeddings as they're too large
                            continue
                        elif key in ["headlines", "entities", "themes"]:
                            # Store as array if it's a list of strings
                            if isinstance(value, list) and all(
                                isinstance(x, str) for x in value
                            ):
                                properties[key] = value[:10]  # Limit to first 10 items
                            else:
                                properties[key] = json.dumps(value)[:1000]
                        else:
                            properties[key] = json.dumps(value)[:1000]
                    elif isinstance(value, str):
                        # Limit string length
                        properties[key] = value[:5000]
                    else:
                        properties[key] = value

                # Create node with label based on type
                node_type = node.get("type", "Unknown")
                if node_type:
                    labels = ["Node", node_type.upper()]
                else:
                    labels = ["Node"]

                query = f"""
                CREATE (n:{":".join(labels)})
                SET n.id = $id,
                    n.type = $type,
                    n += $properties
                """

                session.run(query, id=node["id"], type=node_type, properties=properties)  # type: ignore

                count += 1
                if count % 100 == 0:
                    print(f"  Imported {count}/{len(nodes)} nodes...")

            print(f"✓ Successfully imported {count} nodes!")

    def import_relationships(self, relationships: list):
        """Import relationships into Neo4j."""
        with self.driver.session() as session:
            print(f"\nImporting {len(relationships)} relationships...")
            count = 0

            for rel in relationships:
                # Get relationship type (convert to valid Neo4j relationship type)
                rel_type = rel.get("type", "RELATED_TO").upper().replace(" ", "_")

                # Convert properties
                properties = {}
                for key, value in rel.get("properties", {}).items():
                    if isinstance(value, (list, dict)):
                        # Store overlapped_items as JSON string
                        properties[key] = json.dumps(value)[:1000]
                    elif isinstance(value, str):
                        properties[key] = value[:1000]
                    else:
                        properties[key] = value

                properties["bidirectional"] = rel.get("bidirectional", False)

                query = f"""
                MATCH (source:Node {{id: $source_id}})
                MATCH (target:Node {{id: $target_id}})
                CREATE (source)-[r:{rel_type}]->(target)
                SET r += $properties
                """

                session.run(
                    query,  # type: ignore
                    source_id=rel["source"],
                    target_id=rel["target"],
                    properties=properties,
                )

                count += 1
                if count % 100 == 0:
                    print(f"  Imported {count}/{len(relationships)} relationships...")

            print(f"✓ Successfully imported {count} relationships!")

    def verify_import(self):
        """Verify the import by counting nodes and relationships."""
        with self.driver.session() as session:
            print("\n" + "=" * 60)
            print("VERIFICATION")
            print("=" * 60)

            # Count nodes
            result = session.run("MATCH (n) RETURN count(n) as count")
            node_count = result.single()["count"]
            print(f"Total nodes in Neo4j: {node_count}")

            # Count relationships
            result = session.run("MATCH ()-[r]->() RETURN count(r) as count")
            rel_count = result.single()["count"]
            print(f"Total relationships in Neo4j: {rel_count}")

            # Show node types
            result = session.run("""
                MATCH (n)
                RETURN n.type as type, count(*) as count
                ORDER BY count DESC
            """)
            print("\nNode types:")
            for record in result:
                print(f"  {record['type']}: {record['count']}")

            # Show relationship types
            result = session.run("""
                MATCH ()-[r]->()
                RETURN type(r) as type, count(*) as count
                ORDER BY count DESC
            """)
            print("\nRelationship types:")
            for record in result:
                print(f"  {record['type']}: {record['count']}")

            print("=" * 60)


def load_knowledge_graph(file_path: str) -> Dict[str, Any]:
    """Load knowledge graph from JSON file."""
    print(f"Loading knowledge graph from {file_path}...")
    with open(file_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    print(
        f"✓ Loaded {len(data['nodes'])} nodes and {len(data['relationships'])} relationships"
    )
    return data


def main():
    """Main import function."""
    print("=" * 60)
    print("KNOWLEDGE GRAPH TO NEO4J IMPORTER")
    print("=" * 60)

    # Configuration
    NEO4J_URI = "bolt://localhost:11678"  # Default Neo4j URI
    NEO4J_USERNAME = "neo4j"
    NEO4J_PASSWORD = "password"  # Change this to your Neo4j password

    print(f"\nConnecting to Neo4j at {NEO4J_URI}...")
    print("(Make sure Neo4j is running!)")

    try:
        # Load knowledge graph
        kg_data = load_knowledge_graph("cache/data/kg/pad_17doc_dedup.json")

        # Initialize importer
        importer = Neo4jImporter(NEO4J_URI, NEO4J_USERNAME, NEO4J_PASSWORD)
        print("✓ Connected to Neo4j!")

        # Clear database (optional - comment out if you want to keep existing data)
        clear = input("\nDo you want to clear the existing database? (y/N): ")
        if clear.lower() == "y":
            importer.clear_database()

        # Create constraints
        importer.create_constraints()

        # Import nodes
        importer.import_nodes(kg_data["nodes"])

        # Import relationships
        importer.import_relationships(kg_data["relationships"])

        # Verify import
        importer.verify_import()

        # Close connection
        importer.close()

        print("\n" + "=" * 60)
        print("IMPORT COMPLETED SUCCESSFULLY!")
        print("=" * 60)
        print("\nYou can now query your graph in Neo4j Browser at:")
        print("http://localhost:11474")
        print("\nExample queries:")
        print("  MATCH (n) RETURN n LIMIT 25")
        print("  MATCH (n)-[r]->(m) RETURN n, r, m LIMIT 50")
        print("  MATCH (n:CHUNK) RETURN n LIMIT 10")

    except Exception as e:
        print(f"\n❌ Error: {e}")
        print("\nMake sure:")
        print("1. Neo4j is running (neo4j start)")
        print("2. The password is correct")
        print("3. neo4j Python driver is installed (pip install neo4j)")


if __name__ == "__main__":
    main()
