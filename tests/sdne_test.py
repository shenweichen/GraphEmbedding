import networkx as nx
import tensorflow as tf

from ge import SDNE


def test_SDNE():
    if tf.__version__ >= '1.15.0':
        return #todo
    G = nx.read_edgelist('./tests/Wiki_edgelist.txt',
                         create_using=nx.DiGraph(), nodetype=None, data=[('weight', int)])

    model = SDNE(G, hidden_size=[8, 4], )
    model.train(batch_size=2, epochs=1, verbose=2)
    embeddings = model.get_embeddings()


if __name__ == "__main__":
    pass
