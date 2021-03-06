from scipy.sparse import csr_matrix
import tensorflow as tf
import numpy as np


def graph_to_sparse_tensor(G):
    node2id = {node: i for i, node in enumerate(G.nodes)}
    n_nodes = len(G.nodes)

    indices, values = [], []
    for node in G.nodes:
        neighbors = list(G.neighbors(node))
        norm = 1 / (len(neighbors) + 1)
        node_id = node2id[node]

        values.append(norm)
        indices.append([node_id, node_id])
        for neighbor in neighbors:
            neighbor_id = node2id[neighbor]
            values.append(norm)
            indices.append([node_id, neighbor_id])
    A = tf.SparseTensor(indices, values, (n_nodes, n_nodes))
    A = tf.sparse.reorder(A)
    return A


def graph_to_sparse(G):
    node2id = {node: i for i, node in enumerate(G.nodes)}
    n_nodes = len(G.nodes)

    data, row, col = [], [], []
    for node in G.nodes:
        node_id = node2id[node]
        neighbors = list(G.neighbors(node))
        norm = 1 / (len(neighbors) + 1)

        data.append(norm)
        row.append(node_id)
        col.append(node_id)
        for neighbor in neighbors:
            neighbor_id = node2id[neighbor]
            data.append(norm)
            row.append(node_id)
            col.append(neighbor_id)
    A = csr_matrix((data, (row, col)),
                   shape=(n_nodes, n_nodes),
                   dtype=np.float32)
    return A


def get_agglomerated_features(G, features):
    node2id = {node: i for i, node in enumerate(G.nodes)}

    feats = np.zeros_like(features)
    for node in G.nodes:
        node_id = node2id[node]
        norm = 1 / (G.degree(node) + 1)
        feats[node_id] = features[node_id]
        for neighbor in G.neighbors(node):
            neighbor_id = node2id[neighbor]
            feats[node_id] += features[neighbor_id]
        feats[node_id] *= norm
    return feats
