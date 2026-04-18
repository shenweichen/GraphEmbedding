from pathlib import Path

import networkx as nx
import pytest

pytest.importorskip("gensim")
pytest.importorskip("pandas")
from ge import DeepWalk

TEST_GRAPH_PATH = Path(__file__).resolve().parent / "Wiki_edgelist.txt"


def test_DeepWalk():
    graph = nx.read_edgelist(
        str(TEST_GRAPH_PATH),
        create_using=nx.DiGraph(),
        nodetype=None,
        data=[("weight", int)],
    )

    model = DeepWalk(graph, walk_length=3, num_walks=2, workers=1)
    model.train(embed_size=8, window_size=2, iter=1, workers=1)
    embeddings = model.get_embeddings()
    assert len(embeddings) == graph.number_of_nodes()
    assert all(len(vector) == 8 for vector in embeddings.values())


if __name__ == "__main__":
    pass
