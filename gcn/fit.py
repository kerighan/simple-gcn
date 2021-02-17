from tensorflow.keras.losses import sparse_categorical_crossentropy
import tensorflow as tf
from tqdm import trange


def fit_1_layer(
    A, X, labels, mask, activation,
    n_features, n_labels, latent_dim,
    validation_split, n_epochs
):
    # define tf variables and constants
    W = tf.Variable(tf.random.normal((n_features, latent_dim)))
    O = tf.Variable(tf.random.normal((latent_dim, n_labels)))
    agg_1 = tf.sparse.sparse_dense_matmul(A, X)
    var_list = [W, O]

    # training
    opt = tf.keras.optimizers.Adam(learning_rate=0.1)
    t = trange(n_epochs, desc='training', leave=True)
    for i in t:
        with tf.GradientTape() as tape:
            layer_1 = activation(tf.matmul(agg_1, W))
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
    return out, W, O


def fit_2_layer(
    A, X, labels, mask, activation,
    n_features, n_labels, latent_dim,
    validation_split, n_epochs
):
    # define tf variables and constants
    W_1 = tf.Variable(tf.random.normal((n_features, latent_dim)))
    W_2 = tf.Variable(tf.random.normal((latent_dim, latent_dim)))
    O = tf.Variable(tf.random.normal((latent_dim, n_labels)))
    agg_1 = tf.sparse.sparse_dense_matmul(A, X)
    var_list = [W_1, W_2, O]

    # training
    opt = tf.keras.optimizers.Adam(learning_rate=0.1)
    t = trange(n_epochs, desc='training', leave=True)
    for i in t:
        with tf.GradientTape() as tape:
            layer_1 = activation(tf.matmul(agg_1, W_1))

            agg_2 = tf.sparse.sparse_dense_matmul(A, layer_1)
            layer_2 = activation(tf.matmul(agg_2, W_2))

            out = tf.nn.softmax(tf.matmul(layer_2, O), axis=-1)

            loss = sparse_categorical_crossentropy(labels, out)
            if validation_split > 0:
                loss *= mask
            grad = tape.gradient(loss, var_list)

        opt.apply_gradients(zip(grad, var_list))

        if i % 10 == 0:
            l = loss.numpy().mean()
            t.set_description(f"loss={l:.02}")
            t.refresh()
    return out, W_1, W_2, O
