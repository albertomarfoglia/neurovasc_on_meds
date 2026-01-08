"""
metrics_utils.py

Unified utilities for computing tabular and RDF graph metrics
under a standard output folder structure.

Author: your-name
"""

from pathlib import Path
from typing import Dict, Any, List
from collections import Counter
import statistics
import json

import pandas as pd
from rdflib import Graph, Namespace, RDF, URIRef, BNode, Literal


# =============================================================================
# Tabular metrics (CSV + Parquet)
# =============================================================================

def compute_event_lengths_from_csv(
    csv_path: str,
    event_columns: List[str],
) -> Dict[str, Any]:
    """
    Compute event metrics from syn_data.csv.

    Static events are defined as:
        (#columns in CSV) - (#event columns)
    """
    df = pd.read_csv(csv_path)

    missing = set(event_columns) - set(df.columns)
    if missing:
        raise ValueError(f"Missing event columns in CSV: {missing}")

    event_lengths = {
        event: len(df[df[event] > -1]) for event in event_columns
    }

    n_effective_columns = len(df.columns) - 1  # remove outcome
    static_event_multiplier = n_effective_columns - len(event_columns)
    static_events = len(df) * static_event_multiplier

    return {
        "n_rows": len(df),
        "n_columns": len(df.columns),
        "event_columns": event_columns,
        "static_event_multiplier": static_event_multiplier,
        "event_lengths": event_lengths,
        "static_events": static_events,
        "total_events": sum(event_lengths.values()) + static_events,
    }


def _count_rows_in_parquet_dir(path: Path) -> int:
    """
    Count total rows across all .parquet files in a directory.
    """
    if not path.exists():
        return 0

    total = 0
    for parquet_file in sorted(path.glob("*.parquet")):
        df = pd.read_parquet(parquet_file)
        total += len(df)
    return total


def compute_intermediate_event_metrics(
    static_event_multiplier: int,
) -> Dict[str, Any]:
    """
    Compute event metrics from standard intermediate parquet layout.
    """
    base = Path("intermediate")

    patients = pd.read_parquet(base / "patients.parquet")
    administrations = pd.read_parquet(base / "administrations.parquet")
    procedures = pd.read_parquet(base / "procedures.parquet")

    static_events = len(patients) * static_event_multiplier
    total_events = static_events + len(administrations) + len(procedures)

    return {
        "patients": len(patients),
        "administrations": len(administrations),
        "procedures": len(procedures),
        "static_events": static_events,
        "total_events": total_events,
    }


def compute_split_metrics_from_output(
    output_path: str,
) -> Dict[str, Any]:
    """
    Compute dataset split sizes from standard output/data layout.
    """
    base = Path(output_path) / "data"

    train = _count_rows_in_parquet_dir(base / "train")
    held_out = _count_rows_in_parquet_dir(base / "held_out")
    tuning = _count_rows_in_parquet_dir(base / "tuning")

    return {
        "train": train,
        "held_out": held_out,
        "tuning": tuning,
        "total_events": train + held_out + tuning,
    }


# =============================================================================
# RDF / MEDS graph metrics
# =============================================================================

MEDS = Namespace("https://teamheka.github.io/meds-ontology#")


def _count_node_kinds(g: Graph) -> Dict[str, int]:
    iri_nodes, bnode_nodes, literal_nodes = set(), set(), set()

    for s, p, o in g:
        for node in (s, p, o):
            if isinstance(node, URIRef):
                iri_nodes.add(node)
            elif isinstance(node, BNode):
                bnode_nodes.add(node)
            elif isinstance(node, Literal):
                literal_nodes.add(node)

    return {
        "distinct_iris": len(iri_nodes),
        "distinct_bnodes": len(bnode_nodes),
        "distinct_literals": len(literal_nodes),
    }


def _instances_of(g: Graph, class_uri: URIRef) -> List:
    return list(g.subjects(RDF.type, class_uri))


def _count_recursive_triples(
    g: Graph,
    node,
    visited: set | None = None,
) -> int:
    if visited is None:
        visited = set()

    if node in visited:
        return 0

    visited.add(node)

    count = 0
    for _, _, obj in g.triples((node, None, None)):
        count += 1
        if isinstance(obj, URIRef):
            count += _count_recursive_triples(g, obj, visited)

    return count


def _triples_for_subject(
    g: Graph,
    subject,
    mode: str = "direct",
) -> int:
    """
    mode:
      - 'direct': only subject triples
      - 'recursive': DFS following URIRefs
    """
    if mode == "direct":
        return sum(1 for _ in g.triples((subject, None, None)))

    if mode == "recursive":
        return _count_recursive_triples(g, subject)

    raise ValueError(f"Unknown mode: {mode}")


def compute_graph_stats(
    g: Graph,
    event_triple_mode: str = "direct",
) -> Dict[str, Any]:
    stats: Dict[str, Any] = {}

    stats["total_triples"] = len(g)
    stats["distinct_subjects"] = len(set(g.subjects()))
    stats["distinct_predicates"] = len(set(g.predicates()))
    stats["distinct_objects"] = len(set(g.objects()))

    stats.update(_count_node_kinds(g))

    resources = set(g.subjects()) | {
        o for o in g.objects() if isinstance(o, (URIRef, BNode))
    }
    stats["distinct_resources"] = len(resources)

    meds_classes = {
        "Event": MEDS.Event,
        "Subject": MEDS.Subject,
        "Code": MEDS.Code,
        "LabelSample": MEDS.LabelSample,
        "SubjectSplit": MEDS.SubjectSplit,
        "ValueModality": MEDS.ValueModality,
        "DatasetMetadata": MEDS.DatasetMetadata,
    }

    class_instances = {
        name: _instances_of(g, uri)
        for name, uri in meds_classes.items()
    }

    stats["class_counts"] = {
        name: len(instances)
        for name, instances in class_instances.items()
    }

    event_nodes = class_instances.get("Event", [])
    triples_per_event = [
        _triples_for_subject(g, ev, mode=event_triple_mode)
        for ev in event_nodes
    ]

    stats["n_events"] = len(triples_per_event)
    #stats["triples_per_event_list"] = triples_per_event

    if triples_per_event:
        stats.update({
            "triples_per_event_mean": statistics.mean(triples_per_event),
            "triples_per_event_median": statistics.median(triples_per_event),
            "triples_per_event_min": min(triples_per_event),
            "triples_per_event_max": max(triples_per_event),
            "triples_per_event_pstdev": statistics.pstdev(triples_per_event),
            "triples_per_event_stdev":
                statistics.stdev(triples_per_event)
                if len(triples_per_event) > 1 else 0.0,
        })
    else:
        stats.update({
            "triples_per_event_mean": 0.0,
            "triples_per_event_median": 0.0,
            "triples_per_event_min": 0,
            "triples_per_event_max": 0,
            "triples_per_event_pstdev": 0.0,
            "triples_per_event_stdev": 0.0,
        })

    predicate_counter = Counter(g.predicates())
    stats["distinct_predicate_frequencies"] = {
        str(k): v for k, v in predicate_counter.items()
    }
    stats["top_10_predicates"] = [
        (str(p), c) for p, c in predicate_counter.most_common(10)
    ]

    return stats


# =============================================================================
# Orchestration
# =============================================================================

def collect_all_metrics_from_output(
    *,
    output_path: str,
    csv_path: str,
    graph_path: str,
    event_triple_mode: str = "direct",
) -> Dict[str, Any]:
    """
    Collect all metrics assuming standard folder structure.
    """

    csv_metrics = compute_event_lengths_from_csv(
        csv_path=csv_path,
        event_columns=["paracetamol","nad","corotrop","morphine","dve","atl","iot", "nimodipine"],
    )

    static_event_multiplier = csv_metrics["static_event_multiplier"]

    intermediate_metrics = compute_intermediate_event_metrics(
        static_event_multiplier=static_event_multiplier,
    )

    split_metrics = compute_split_metrics_from_output(output_path)

    g = Graph()
    g.parse(graph_path, format="turtle")

    graph_metrics = compute_graph_stats(
        g,
        event_triple_mode=event_triple_mode,
    )

    consistency_checks = {
        "csv_vs_intermediate_match":
            csv_metrics["total_events"] == intermediate_metrics["total_events"],
        "intermediate_vs_splits_match":
            intermediate_metrics["total_events"] == split_metrics["total_events"],
        "graph_event_count_match":
            intermediate_metrics["total_events"]
            == graph_metrics["class_counts"]["Event"],
    }

    derived_metrics = {
        "events_per_patient":
            intermediate_metrics["total_events"]
            / intermediate_metrics["patients"],
        "avg_triples_per_event":
            graph_metrics["triples_per_event_mean"],
        "graph_density":
            graph_metrics["total_triples"]
            / graph_metrics["distinct_resources"],
    }

    return {
        "tabular_metrics": {
            "csv": csv_metrics,
            "intermediate": intermediate_metrics,
            "splits": split_metrics,
        },
        "graph_metrics": graph_metrics,
        "consistency_checks": consistency_checks,
        "derived_metrics": derived_metrics,
    }


# =============================================================================
# Serialization
# =============================================================================

def save_stats_json(stats: Dict[str, Any], path: str) -> None:
    """
    Save metrics dict as formatted JSON.
    Creates parent directories if they do not exist.
    """
    ppath = Path(path)

    # Ensure parent directory exists
    if ppath.parent:
        ppath.parent.mkdir(parents=True, exist_ok=True)

    with ppath.open("w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2)
