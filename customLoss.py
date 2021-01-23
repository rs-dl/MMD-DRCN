import tensorflow as tf
import keras.backend as K


def huber_loss(y_true, y_pred):
    return tf.losses.huber_loss(y_true, y_pred)


def adjust_binary_cross_entropy(y_true, y_pred):
    return K.binary_crossentropy(y_true, K.pow(y_pred, 2))


def MMD_Loss_func(num_source, sigmas=None):
    if sigmas is None:
        sigmas = [1, 5, 10]

    def loss(y_true, y_pred):
        cost = []

        for i in range(num_source):
            for j in range(num_source):
                domain_i = tf.where(tf.equal(y_true, i))[:, 0]
                domain_j = tf.where(tf.equal(y_true, j))[:, 0]
                single_res = mmd_two_distribution(K.gather(y_pred, domain_i),
                                                  K.gather(y_pred, domain_j),
                                                  sigmas=sigmas)
                cost.append(single_res)
        #print("wtf")
        cost = K.concatenate(cost)
        return K.mean(cost)
    return loss


def mmd_two_distribution(source, target, sigmas):

    sigmas = K.constant(sigmas)
    xy = rbf_kernel(source, target, sigmas)
    xx = rbf_kernel(source, source, sigmas)
    yy = rbf_kernel(target, target, sigmas)
    return xx + yy - 2 * xy


def rbf_kernel(x, y, sigmas):
    beta = 1. / (2. * (tf.expand_dims(sigmas, 1)))
    dist = compute_pairwise_distances(x, y)
    dot = -K.dot(beta, K.reshape(dist, (1, -1)))
    exp = K.exp(dot)
    return K.mean(exp, keepdims=True)


def compute_pairwise_distances(x, y):
    norm = lambda x: K.sum(K.square(x), axis=1)
    return norm(K.expand_dims(x, 2) - K.transpose(y))
