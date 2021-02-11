from tensorflow.keras.losses import sparse_categorical_crossentropy
from .utils import graph_to_sparse, graph_to_sparse_tensor
from tqdm import trange
import tensorflow as tf
import networkx as nx
import numpy as np


class GCN:
    def __init__(
        self,
        latent_dim=48,
        activation="sigmoid",
        n_epochs=300,
        lr=0.001,
        decay=1.,
        validation_split=.05
    ):
        self.latent_dim = latent_dim
        self.activation = activation
        self.n_epochs = n_epochs
        self.lr = lr
        self.decay = decay
        self.validation_split = validation_split

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
        lr = self.lr
        t = trange(self.n_epochs, desc='training', leave=True)
        for i in t:
            with tf.GradientTape() as tape:
                layer_1 = activation(tf.matmul(agg_1, W_1))
                out = tf.nn.softmax(tf.matmul(layer_1, O), axis=-1)

                loss = sparse_categorical_crossentropy(labels, out)
                if validation_split > 0:
                    loss *= mask
                dL_dW_1, dL_dO = tape.gradient(loss, [W_1, O])
            W_1.assign_sub(lr * dL_dW_1)
            O.assign_sub(lr * dL_dO)

            lr *= self.decay
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
