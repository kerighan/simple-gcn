from gcn import GCN
import networkx as nx
import numpy as np

# load CORA dataset
G = nx.read_gexf("datasets/G.gexf")
features = np.load("datasets/features.npy")
labels = np.load("datasets/labels.npy").argmax(axis=1)

# create GCN
gcn = GCN(n_layers=3, activation="tanh", validation_split=0.5)

# fit and persist model to disk
gcn.fit(G, features, labels)
gcn.save("test.p")

# load and predict
gcn = GCN.load("test.p")
y_pred = gcn.predict(G, features)
print(y_pred)
print(np.mean(labels == y_pred))
