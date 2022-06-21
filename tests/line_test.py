import networkx as nx

from ge import LINE


def test_LINE():
    G = nx.read_edgelist('./tests/Wiki_edgelist.txt',
                         create_using=nx.DiGraph(), nodetype=None, data=[('weight', int)])

    model = LINE(G, embedding_size=2, order='second')
    model.train(batch_size=2, epochs=1, verbose=2)
    embeddings = model.get_embeddings()


if __name__ == "__main__":
    pass
