from pathlib import Path

import networkx as nx
import pytest

pytest.importorskip("tensorflow")
from ge import LINE

TEST_GRAPH_PATH = Path(__file__).resolve().parent / "Wiki_edgelist.txt"


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
