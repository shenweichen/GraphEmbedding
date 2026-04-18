from pathlib import Path

import networkx as nx
import pytest

pytest.importorskip("tensorflow")
from ge import SDNE

TEST_GRAPH_PATH = Path(__file__).resolve().parent / "Wiki_edgelist.txt"


def test_SDNE():
    graph = nx.read_edgelist(
        str(TEST_GRAPH_PATH),
        create_using=nx.DiGraph(),
        nodetype=None,
        data=[("weight", int)],
    )

    model = SDNE(graph, hidden_size=[8, 4])
    model.train(batch_size=2, epochs=1, verbose=0)
    embeddings = model.get_embeddings()
    assert len(embeddings) == graph.number_of_nodes()
    assert all(len(vector) == 4 for vector in embeddings.values())


if __name__ == "__main__":
    pass
