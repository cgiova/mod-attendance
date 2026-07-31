"""Modality-attendance building blocks shared by the training and post-hoc notebooks.

The 5 custom ``keras`` layers below are the mechanisms behind the modality-attendance
strategies (fixed / learnable scaling, controlled fusion, adaptive gating). They are
``@register_keras_serializable(package="modality_attendance")`` so that
``keras.models.load_model`` can deserialize every strategy checkpoint **as long as this
module has been imported** (the decorators register the classes at import time).

They used to be declared inline in the framework notebook in two places (the training
builder cells *and* a re-registration cell in the post-hoc half). Keeping a single copy
here means the training notebook and the post-hoc notebook cannot drift out of sync — if
they did, a checkpoint trained by one could silently fail to load in the other, the exact
deserialization failure the FeatureSpace guard exists to catch.

The layers are pure Keras/NumPy/TF and dataset-agnostic (no notebook globals), so this
module also serves the BRSET notebook and any further datasets.

Backend note: import ``os`` and set ``KERAS_BACKEND=tensorflow`` (as the notebooks do in
their first cell) *before* importing this module, so ``keras`` binds to the TF backend.
"""

import numpy as np
import tensorflow as tf
import keras
import keras.ops as K


# --------------------------------------------------------------------------- #
# Custom serializable layers (the modality-attendance mechanisms)
# --------------------------------------------------------------------------- #
@keras.utils.register_keras_serializable(package="modality_attendance")
class ConstantScale(keras.layers.Layer):
    """Multiply an activation by a fixed (non-trainable) scalar. Used by fixed scaling."""

    def __init__(self, scale, **kwargs):
        super().__init__(**kwargs)
        self.scale = float(scale)

    def call(self, x):
        return x * keras.ops.cast(self.scale, x.dtype)

    def get_config(self):
        cfg = super().get_config()
        cfg.update({"scale": self.scale})
        return cfg


@keras.utils.register_keras_serializable(package="modality_attendance")
class LearnableAlpha(keras.layers.Layer):
    """Learn a scalar split ``(alpha, beta=1-alpha)`` via a sigmoid on a trainable logit."""

    def __init__(self, init_alpha=0.5, **kwargs):
        super().__init__(**kwargs)
        init_alpha = float(np.clip(init_alpha, 1e-4, 1.0 - 1e-4))
        self.init_alpha = init_alpha
        init_logit = np.log(init_alpha / (1.0 - init_alpha))
        self.logit = self.add_weight(
            name="alpha_logit",
            shape=(),
            initializer=keras.initializers.Constant(init_logit),
            trainable=True,
        )

    def call(self, x):
        alpha = tf.sigmoid(self.logit)
        beta = 1.0 - alpha
        return alpha, beta

    def get_config(self):
        cfg = super().get_config()
        cfg.update({"init_alpha": self.init_alpha})
        return cfg


@keras.utils.register_keras_serializable(package="modality_attendance")
class ScalarMultiply(keras.layers.Layer):
    """Elementwise-broadcast multiply ``x * s`` for a tensor ``x`` and scalar ``s``."""

    def call(self, inputs):
        x, s = inputs
        s = keras.ops.cast(s, x.dtype)
        return x * s

    def compute_output_shape(self, input_shape):
        return input_shape[0]


@keras.utils.register_keras_serializable(package="modality_attendance")
class FusionWithGamma(keras.layers.Layer):
    """Weighted sum ``gamma_img*z_img + gamma_tab*z_tab`` with non-trainable gammas.

    Gammas are stored as (frozen) weights so they can be adjusted post-training to
    re-bias the fusion without retraining.
    """

    def __init__(self, gamma_img=1.0, gamma_tab=1.0, **kwargs):
        super().__init__(**kwargs)
        self.initial_gamma_img = float(gamma_img)
        self.initial_gamma_tab = float(gamma_tab)
        self.gamma_img = self.add_weight(
            name="gamma_img", shape=(),
            initializer=keras.initializers.Constant(self.initial_gamma_img),
            trainable=False,
        )
        self.gamma_tab = self.add_weight(
            name="gamma_tab", shape=(),
            initializer=keras.initializers.Constant(self.initial_gamma_tab),
            trainable=False,
        )

    def call(self, inputs):
        z_img, z_tab = inputs
        return self.gamma_img * z_img + self.gamma_tab * z_tab

    def get_config(self):
        cfg = super().get_config()
        cfg.update({
            "gamma_img": self.initial_gamma_img,
            "gamma_tab": self.initial_gamma_tab,
        })
        return cfg


@keras.utils.register_keras_serializable(package="modality_attendance")
class GateEntropyLoss(keras.layers.Layer):
    """Pass-through layer that adds a negative-entropy penalty on the softmax gate.

    Maximizes gate entropy (weighted by ``weight``) to stop the gate from collapsing to a
    deterministic single-modality choice.
    """

    def __init__(self, weight=1e-4, **kwargs):
        super().__init__(**kwargs)
        self.weight = float(weight)

    def call(self, gate_out):
        entropy = -K.mean(K.sum(gate_out * K.log(gate_out + 1e-6), axis=1))
        self.add_loss(-self.weight * entropy)  # maximize entropy
        return gate_out

    def get_config(self):
        cfg = super().get_config()
        cfg.update({"weight": self.weight})
        return cfg


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
