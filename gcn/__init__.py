from tensorflow.keras.losses import sparse_categorical_crossentropy
from .utils import (
    graph_to_sparse,
    graph_to_sparse_tensor,
    get_agglomerated_features)
from tqdm import trange
import tensorflow as tf
import networkx as nx
import numpy as np


class GCN:
    def __init__(
        self,
        latent_dim=48,
        activation="sigmoid",
        n_epochs=100,
        validation_split=.05,
        level="node"
    ):
        self.latent_dim = latent_dim
        self.activation = activation
        self.n_epochs = n_epochs
        self.validation_split = validation_split
        self.level = level

    @staticmethod
    def load(filename):
        import pickle
        with open(filename, "rb") as f:
            obj = pickle.load(f)
        return obj

    def save(self, filename):
        import pickle
        with open(filename, "wb") as f:
            pickle.dump(self, f)

    @staticmethod
    def get_graphs_features(data):
        feats = []
        max_len = 0
        for G, features in data:
            features = get_agglomerated_features(G, features)
            if features.shape[0] > max_len:
                max_len = features.shape[0]
            feats.append(features)
            dim = features.shape[1]
        
        features = np.zeros((len(data), max_len, dim))
        for i, f in enumerate(feats):
            features[i, :f.shape[0]] = f
        return features

    def fit_graphs(self, features, labels):
        input_length = features.shape[1]
        input_dim = features.shape[2]
        n_labels = labels.max() + 1

        X = tf.constant(features, dtype=np.float32)
        W = tf.Variable(tf.random.normal((input_dim, self.latent_dim)))
        D = tf.Variable(tf.random.normal((self.latent_dim, self.latent_dim // 2)))
        S = tf.Variable(tf.random.normal((self.latent_dim, 1)))
        O = tf.Variable(tf.random.normal((self.latent_dim // 2, n_labels)))
        scalar = tf.Variable(25.)
        var_list = [W, S, O, D, scalar]

        t = trange(self.n_epochs, desc='training', leave=True)
        opt = tf.keras.optimizers.Adam(learning_rate=0.1)
        for i in t:
            with tf.GradientTape() as tape:
                values = tf.matmul(X, W)

                score = scalar * tf.tanh(tf.matmul(values, S))
                score = tf.nn.softmax(score, axis=-2)
                embedding = tf.math.reduce_sum(score * values, axis=1, keepdims=False)

                dense = tf.nn.sigmoid(tf.matmul(embedding, D))
                out = tf.nn.softmax(tf.matmul(dense, O), axis=-1)
                loss = tf.keras.losses.sparse_categorical_crossentropy(labels, out)

                grad = tape.gradient(loss, var_list)

            opt.apply_gradients(zip(grad, var_list))

            if i % 10 == 0:
                l = loss.numpy().mean()
                t.set_description(f"loss={l:.02}")
                t.refresh()
        
        y_pred = out.numpy().argmax(axis=-1)
        accuracy = (y_pred == labels).mean()
        print(f"accuracy={accuracy:.02}")

    def fit(self, G, features, labels):
        # define parameters
        n_nodes, n_features = features.shape
        n_labels = labels.max() + 1

        # define training samples
        validation_split = self.validation_split
        if validation_split > 0:
            val_size = int(round(n_nodes * validation_split))
            index_train = np.random.choice(
                range(n_nodes), size=val_size, replace=False)
            mask = np.ones((n_nodes,), dtype=np.float32)
            mask[index_train] = 0

        # define tf variables and constants
        latent_dim = self.latent_dim
        A = graph_to_sparse_tensor(G)
        X = tf.constant(features, dtype=np.float32)
        W_1 = tf.Variable(tf.random.normal((n_features, latent_dim)))
        O = tf.Variable(tf.random.normal((latent_dim, n_labels)))
        agg_1 = tf.sparse.sparse_dense_matmul(A, X)
        var_list = [W_1, O]

        # define activation function
        activation = self.activation
        if activation == "sigmoid":
            activation = tf.nn.sigmoid
        elif activation == "tanh":
            activation = tf.nn.tanh
        elif activation == "relu":
            activation = tf.nn.relu
        else:
            activation = tf.keras.activations.linear

        # training
        opt = tf.keras.optimizers.Adam(learning_rate=0.1)
        t = trange(self.n_epochs, desc='training', leave=True)
        for i in t:
            with tf.GradientTape() as tape:
                layer_1 = activation(tf.matmul(agg_1, W_1))
                out = tf.nn.softmax(tf.matmul(layer_1, O), axis=-1)

                loss = sparse_categorical_crossentropy(labels, out)
                if validation_split > 0:
                    loss *= mask
                grad = tape.gradient(loss, var_list)

            opt.apply_gradients(zip(grad, var_list))

            if i % 10 == 0:
                l = loss.numpy().mean()
                t.set_description(f"loss={l:.02}")
                t.refresh()

        y_pred = out.numpy().argmax(axis=1)
        accuracy = np.mean(y_pred == labels)
        print(f"accuracy={accuracy:.02}")
        if validation_split > 0:
            val_accuracy = np.mean(y_pred[index_train] == labels[index_train])
            print(f"val_accuracy={val_accuracy:.02}")
        
        self.W = W_1.numpy()
        self.O = O.numpy()
    
    def predict(self, G, features):
        # define parameters
        n_nodes, n_features = features.shape

        # define activation function
        activation = self.activation
        if activation == "sigmoid":
            activation = lambda x: 1 / (1 + np.exp(-x))
        elif activation == "tanh":
            activation = np.tanh
        elif activation == "relu":
            activation = lambda x: np.maximum(x, 0)
        else:
            activation = lambda x: x

        # get matrices
        A = graph_to_sparse(G)
        agg = A @ features
        layer_1 = activation(agg @ self.W)
        out = softmax(layer_1 @ self.O)
        return out.argmax(axis=1)


def softmax(x):
    score = np.exp(x)
    score /= score.sum(axis=1)
    return score
