# -*- coding:utf-8 -*-

"""



Author:

    Weichen Shen,weichenswc@163.com



Reference:

    [1] Wang D, Cui P, Zhu W. Structural deep network embedding[C]//Proceedings of the 22nd ACM SIGKDD international conference on Knowledge discovery and data mining. ACM, 2016: 1225-1234.(https://www.kdd.org/kdd2016/papers/files/rfp0191-wangAemb.pdf)



"""
import time

import numpy as np
import scipy.sparse as sp
import tensorflow as tf
from tensorflow.keras.callbacks import History
from tensorflow.keras.layers import Dense, Input
from tensorflow.keras.models import Model
from tensorflow.keras.regularizers import l1_l2

from ..utils import preprocess_nxgraph


def l_2nd(beta):
    def loss_2nd(y_true, y_pred):
        beta_weight = tf.cast(beta, y_true.dtype)
        ones = tf.ones_like(y_true)
        b_ = tf.where(tf.not_equal(y_true, 0), beta_weight * ones, ones)
        x = tf.square((y_true - y_pred) * b_)
        return tf.reduce_mean(tf.reduce_sum(x, axis=-1))

    return loss_2nd


def l_1st(alpha):
    def loss_1st(y_true, y_pred):
        laplacian = y_true
        embeddings = y_pred
        batch_size = tf.cast(tf.shape(laplacian)[0], embeddings.dtype)
        alpha_weight = tf.cast(alpha, embeddings.dtype)
        return (
            alpha_weight
            * 2.0
            * tf.linalg.trace(tf.matmul(tf.matmul(embeddings, laplacian, transpose_a=True), embeddings))
            / batch_size
        )

    return loss_1st


def create_model(node_size, hidden_size=[256, 128], l1=1e-5, l2=1e-4):
    A = Input(shape=(node_size,))
    fc = A
    for i in range(len(hidden_size)):
        if i == len(hidden_size) - 1:
            fc = Dense(hidden_size[i], activation='relu',
                       kernel_regularizer=l1_l2(l1, l2), name='1st')(fc)
        else:
            fc = Dense(hidden_size[i], activation='relu',
                       kernel_regularizer=l1_l2(l1, l2))(fc)
    Y = fc
    for i in reversed(range(len(hidden_size) - 1)):
        fc = Dense(hidden_size[i], activation='relu',
                   kernel_regularizer=l1_l2(l1, l2))(fc)

    A_ = Dense(node_size, 'relu', name='2nd')(fc)
    model = Model(inputs=A, outputs=[A_, Y])
    emb = Model(inputs=A, outputs=Y)
    return model, emb


class SDNE(object):
    def __init__(self, graph, hidden_size=[32, 16], alpha=1e-6, beta=5., nu1=1e-5, nu2=1e-4, ):

        self.graph = graph
        # self.g.remove_edges_from(self.g.selfloop_edges())
        self.idx2node, self.node2idx = preprocess_nxgraph(self.graph)

        self.node_size = self.graph.number_of_nodes()
        self.hidden_size = hidden_size
        self.alpha = alpha
        self.beta = beta
        self.nu1 = nu1
        self.nu2 = nu2

        self.A, self.L = _create_A_L(self.graph, self.node2idx)  # Adj Matrix,L Matrix
        self.reset_model()
        self._embeddings = {}

    def reset_model(self, opt='adam'):

        self.model, self.emb_model = create_model(self.node_size, hidden_size=self.hidden_size, l1=self.nu1,
                                                  l2=self.nu2)
        self.model.compile(opt, [l_2nd(self.beta), l_1st(self.alpha)])
        self.get_embeddings()

    def train(self, batch_size=1024, epochs=1, initial_epoch=0, verbose=1):
        adjacency = self.A.toarray().astype(np.float32)
        laplacian = self.L.toarray().astype(np.float32)
        if batch_size >= self.node_size:
            if batch_size > self.node_size:
                print('batch_size({0}) > node_size({1}),set batch_size = {1}'.format(
                    batch_size, self.node_size))
                batch_size = self.node_size
            return self.model.fit(
                adjacency,
                [adjacency, laplacian],
                batch_size=batch_size,
                epochs=epochs,
                initial_epoch=initial_epoch,
                verbose=verbose,
                shuffle=False,
            )
        else:
            steps_per_epoch = (self.node_size - 1) // batch_size + 1
            hist = History()
            hist.set_model(self.model)
            hist.on_train_begin()
            logs = {}
            for epoch in range(initial_epoch, epochs):
                start_time = time.time()
                losses = np.zeros(3)
                for i in range(steps_per_epoch):
                    index = np.arange(
                        i * batch_size, min((i + 1) * batch_size, self.node_size))
                    A_train = adjacency[index, :]
                    L_mat_train = laplacian[index][:, index]
                    batch_losses = np.asarray(self.model.train_on_batch(A_train, [A_train, L_mat_train]))
                    losses += batch_losses
                losses = losses / steps_per_epoch

                logs['loss'] = losses[0]
                logs['2nd_loss'] = losses[1]
                logs['1st_loss'] = losses[2]
                epoch_time = int(time.time() - start_time)
                hist.on_epoch_end(epoch, logs)
                if verbose > 0:
                    print('Epoch {0}/{1}'.format(epoch + 1, epochs))
                    print('{0}s - loss: {1: .4f} - 2nd_loss: {2: .4f} - 1st_loss: {3: .4f}'.format(
                        epoch_time, losses[0], losses[1], losses[2]))
            return hist

    def evaluate(self, ):
        adjacency = self.A.toarray().astype(np.float32)
        laplacian = self.L.toarray().astype(np.float32)
        return self.model.evaluate(x=adjacency, y=[adjacency, laplacian], batch_size=self.node_size)

    def get_embeddings(self):
        self._embeddings = {}
        adjacency = self.A.toarray().astype(np.float32)
        embeddings = self.emb_model.predict(adjacency, batch_size=self.node_size, verbose=0)
        look_back = self.idx2node
        for i, embedding in enumerate(embeddings):
            self._embeddings[look_back[i]] = embedding

        return self._embeddings


def _create_A_L(graph, node2idx):
    node_size = graph.number_of_nodes()
    A_data = []
    A_row_index = []
    A_col_index = []

    for edge in graph.edges():
        v1, v2 = edge
        edge_weight = graph[v1][v2].get('weight', 1)

        A_data.append(edge_weight)
        A_row_index.append(node2idx[v1])
        A_col_index.append(node2idx[v2])

    A = sp.csr_matrix((A_data, (A_row_index, A_col_index)), shape=(node_size, node_size))
    A_ = sp.csr_matrix((A_data + A_data, (A_row_index + A_col_index, A_col_index + A_row_index)),
                       shape=(node_size, node_size))

    D = sp.diags(A_.sum(axis=1).flatten().tolist()[0])
    L = D - A_
    return A, L
