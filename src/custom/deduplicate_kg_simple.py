#!/usr/bin/env python3
"""
Simple Knowledge Graph Deduplication Script

This script removes duplicate nodes based on ID only.
If multiple nodes have the same ID, keeps the first one and removes the rest.
"""

import argparse
import json
from pathlib import Path
from typing import Optional


def deduplicate_by_id(
    input_file: Path, output_file: Optional[Path] = None, backup: bool = True
) -> None:
    """
    Remove duplicate nodes based on ID only.

    Args:
        input_file: Path to input JSON file
        output_file: Path to output JSON file (defaults to input_file)
        backup: Whether to create a backup of the original file
    """
    if output_file is None:
        output_file = input_file

    print(f"Loading knowledge graph from: {input_file}")

    # Load the knowledge graph
    with open(input_file, "r", encoding="utf-8") as f:
        kg_data = json.load(f)

    nodes = kg_data.get("nodes", [])
    relationships = kg_data.get("relationships", [])

    print(f"Original: {len(nodes)} nodes, {len(relationships)} relationships")

    # Find and remove duplicate nodes by ID
    seen_ids = set()
    unique_nodes = []
    removed_count = 0

    for node in nodes:
        node_id = node["id"]
        if node_id not in seen_ids:
            seen_ids.add(node_id)
            unique_nodes.append(node)
            print(f"✅ Keeping: {node_id}")
        else:
            removed_count += 1
            print(f"❌ Removing duplicate: {node_id}")

    print("\nDeduplication summary:")
    print(f"  Original nodes: {len(nodes)}")
    print(f"  Removed duplicates: {removed_count}")
    print(f"  Final nodes: {len(unique_nodes)}")

    # Create backup if requested
    if backup and output_file == input_file:
        backup_file = input_file.with_suffix(".backup.json")
        print(f"\nCreating backup: {backup_file}")
        with open(backup_file, "w", encoding="utf-8") as f:
            json.dump(kg_data, f, indent=2, ensure_ascii=False)

    # Save deduplicated knowledge graph
    deduplicated_kg = {"nodes": unique_nodes, "relationships": relationships}

    print(f"\nSaving deduplicated knowledge graph to: {output_file}")
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(deduplicated_kg, f, indent=2, ensure_ascii=False)

    print("✅ Deduplication completed successfully!")


def main():
    parser = argparse.ArgumentParser(
        description="Remove duplicate nodes by ID from knowledge graph JSON file"
    )
    parser.add_argument("input_file", type=Path, help="Input knowledge graph JSON file")
    parser.add_argument(
        "-o", "--output", type=Path, help="Output file (defaults to input file)"
    )
    parser.add_argument(
        "--no-backup", action="store_true", help="Skip creating backup file"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be done without making changes",
    )

    args = parser.parse_args()

    if not args.input_file.exists():
        print(f"Error: Input file {args.input_file} does not exist")
        return 1

    if args.dry_run:
        print("🔍 DRY RUN MODE - No changes will be made")

        # Load and analyze without making changes
        with open(args.input_file, "r", encoding="utf-8") as f:
            kg_data = json.load(f)

        nodes = kg_data.get("nodes", [])
        seen_ids = set()
        duplicate_count = 0

        for node in nodes:
            node_id = node["id"]
            if node_id in seen_ids:
                duplicate_count += 1
                print(f"❌ Would remove duplicate: {node_id}")
            else:
                seen_ids.add(node_id)

        print(f"\nWould remove {duplicate_count} duplicate nodes")
    else:
        deduplicate_by_id(args.input_file, args.output, backup=not args.no_backup)

    return 0


if __name__ == "__main__":
    exit(main())
