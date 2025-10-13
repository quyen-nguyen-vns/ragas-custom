import json
from pathlib import Path

dataset_dir = "cache/data/dataset"


def aggregate_results(list_results_files: list[str]) -> list:
    """Aggregate results from multiple files."""
    results = []
    for file in list_results_files:
        with open(Path(dataset_dir) / file, "r") as f:
            results.extend(json.load(f))
    return results


if __name__ == "__main__":
    list_results_files = [
        "pad_17doc_100_0.json",
        "pad_17doc_100_1.json",
        "pad_17doc_100_2.json",
        "pad_17doc_100_3.json",
        "pad_17doc_100_4.json",
        "pad_17doc_100_5.json",
        "pad_17doc_perfect_grammar_100_0.json",
        "pad_17doc_perfect_grammar_th_100_0.json",
    ]
    results = aggregate_results(list_results_files)

    # save results to json - allow to save thai characters
    name = "eval_set_v0.json"
    with open(Path(dataset_dir) / name, "w") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    # show number of data
    print(f"Number of data: {len(results)}")
