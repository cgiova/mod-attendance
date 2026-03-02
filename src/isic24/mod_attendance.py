def weighted_loss_ps(y_true, y_pred):
    """
    Per-sample weighted regularized loss
    :param y_true:
    :param y_pred:
    :return:
    """
    # BCE base
    base_loss = keras.losses.binary_crossentropy(y_true, y_pred, label_smoothing=0.02)

    # Penalità sui pesi che connettono la parte tabellare all’output
    W = dense_final.kernel  # matrice [dim_img + dim_tab, 1]
    W_tab = W[dim_img:]     # seleziona solo la parte tabellare
    reg_term = lambda_reg * tf.reduce_sum(tf.square(W_tab))

    return base_loss + reg_term


def weighted_loss_pb(y_true, y_pred):
    """
    Per-batch weighted regularized loss
    :param y_true:
    :param y_pred:
    :return:
    """
    base_loss = keras.losses.binary_crossentropy(
        y_true, y_pred, label_smoothing=0.02
    )
    base_loss = tf.reduce_mean(base_loss)  # per-batch mean
    W = dense_final.kernel
    W_tab = W[dim_img:]
    reg_term = lambda_reg * tf.reduce_sum(tf.square(W_tab))
    return base_loss + reg_term