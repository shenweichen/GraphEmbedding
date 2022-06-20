

from ge import DeepWalk
import networkx as nx


if __name__ == "__main__":
    G = nx.read_edgelist('./Wiki_edgelist.txt',
                         create_using=nx.DiGraph(), nodetype=None, data=[('weight', int)])

    model = DeepWalk(G, walk_length=3, num_walks=2, workers=1)
    model.train(window_size=3, iter=1)
    embeddings = model.get_embeddings()