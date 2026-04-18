from pathlib import Path
import tempfile

import networkx as nx
import pytest

pytest.importorskip("fastdtw")
pytest.importorskip("gensim")
pytest.importorskip("pandas")
from ge import Struc2Vec

TEST_GRAPH_PATH = Path(__file__).resolve().parent / "Wiki_edgelist.txt"


def test_Struc2Vec():
    graph = nx.read_edgelist(
        str(TEST_GRAPH_PATH),
        create_using=nx.DiGraph(),
        nodetype=None,
        data=[("weight", int)],
    )

    with tempfile.TemporaryDirectory(prefix="struc2vec-test-") as temp_dir:
        model = Struc2Vec(
            graph,
            walk_length=3,
            num_walks=1,
            workers=1,
            verbose=0,
            temp_path=temp_dir + "/",
        )
        model.train(embed_size=8, window_size=2, workers=1, iter=1)
        embeddings = model.get_embeddings()
    assert len(embeddings) == graph.number_of_nodes()
    assert all(len(vector) == 8 for vector in embeddings.values())


if __name__ == "__main__":
    pass
