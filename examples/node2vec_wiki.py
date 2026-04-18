from pathlib import Path
import sys

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.manifold import TSNE

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from ge import Node2Vec
from ge.classify import Classifier, read_node_label

WIKI_GRAPH_PATH = PROJECT_ROOT / "data" / "wiki" / "Wiki_edgelist.txt"
WIKI_LABEL_PATH = PROJECT_ROOT / "data" / "wiki" / "wiki_labels.txt"
SMOKE_GRAPH_PATH = PROJECT_ROOT / "tests" / "Wiki_edgelist.txt"


def evaluate_embeddings(embeddings, label_path):
    x_data, y_data = read_node_label(str(label_path))
    train_fraction = 0.8
    print("Training classifier using {:.2f}% nodes...".format(train_fraction * 100))
    clf = Classifier(embeddings=embeddings, clf=LogisticRegression())
    clf.split_train_evaluate(x_data, y_data, train_fraction)


def plot_embeddings(embeddings, label_path, show=True):
    x_data, y_data = read_node_label(str(label_path))

    embedding_list = np.array([embeddings[node] for node in x_data])
    node_pos = TSNE(n_components=2).fit_transform(embedding_list)

    color_idx = {}
    for index, label in enumerate(y_data):
        color_idx.setdefault(label[0], [])
        color_idx[label[0]].append(index)

    for label, indexes in color_idx.items():
        plt.scatter(node_pos[indexes, 0], node_pos[indexes, 1], label=label)
    plt.legend()
    if show:
        plt.show()
    else:
        plt.close()


def main(smoke=False, show=True):
    graph_path = SMOKE_GRAPH_PATH if smoke else WIKI_GRAPH_PATH
    graph = nx.read_edgelist(
        str(graph_path),
        create_using=nx.DiGraph(),
        nodetype=None,
        data=[("weight", int)],
    )

    model = Node2Vec(
        graph,
        walk_length=3 if smoke else 10,
        num_walks=2 if smoke else 80,
        p=0.25,
        q=4,
        workers=1,
        use_rejection_sampling=False,
    )
    model.train(window_size=2 if smoke else 5, iter=1 if smoke else 3, workers=1)
    embeddings = model.get_embeddings()
    assert len(embeddings) > 0

    if not smoke:
        evaluate_embeddings(embeddings, WIKI_LABEL_PATH)
        plot_embeddings(embeddings, WIKI_LABEL_PATH, show=show)

    return embeddings


if __name__ == "__main__":
    main()
