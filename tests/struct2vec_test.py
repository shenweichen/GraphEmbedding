import networkx as nx

from ge import Struc2Vec


def test_Struc2Vec():
    G = nx.read_edgelist('./tests/Wiki_edgelist.txt', create_using=nx.DiGraph(), nodetype=None,
                         data=[('weight', int)])

    model = Struc2Vec(G, 3, 1, workers=1, verbose=40, )
    model.train()
    embeddings = model.get_embeddings()


if __name__ == "__main__":
    pass
