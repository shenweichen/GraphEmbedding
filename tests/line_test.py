from pathlib import Path

import networkx as nx
import pytest

pytest.importorskip("tensorflow")
from ge import LINE
from ge.models import line as line_module
from ge.utils import preprocess_nxgraph

TEST_GRAPH_PATH = Path(__file__).resolve().parent / "Wiki_edgelist.txt"


def test_LINE_sampling_tables_use_normalized_probabilities(monkeypatch):
    graph = nx.DiGraph()
    graph.add_edge("a", "b", weight=2)
    graph.add_edge("a", "c", weight=6)

    model = LINE.__new__(LINE)
    model.graph = graph
    model.idx2node, model.node2idx = preprocess_nxgraph(graph)
    model.node_size = graph.number_of_nodes()

    sampling_tables = []

    def record_sampling_table(area_ratio):
        sampling_tables.append(list(area_ratio))
        return [1] * len(area_ratio), [0] * len(area_ratio)

    monkeypatch.setattr(line_module, "create_alias_table", record_sampling_table)

    LINE._gen_sampling_table(model)

    node_probs, edge_probs = sampling_tables
    expected_edge_probs = [
        graph[edge[0]][edge[1]]["weight"] / 8 for edge in graph.edges()
    ]

    assert sum(node_probs) == pytest.approx(1)
    assert sum(edge_probs) == pytest.approx(1)
    assert edge_probs == pytest.approx(expected_edge_probs)


def test_LINE():
    graph = nx.read_edgelist(
        str(TEST_GRAPH_PATH),
        create_using=nx.DiGraph(),
        nodetype=None,
        data=[("weight", int)],
    )

    model = LINE(graph, embedding_size=4, order="second")
    model.train(batch_size=2, epochs=1, verbose=0)
    embeddings = model.get_embeddings()
    assert len(embeddings) == graph.number_of_nodes()
    assert all(len(vector) == 4 for vector in embeddings.values())


if __name__ == "__main__":
    pass
