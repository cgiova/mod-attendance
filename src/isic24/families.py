"""
The six modality-attendance families: one builder each, assembled from `mod_attendance`.
"""

from dataclasses import dataclass
from typing import Any

import keras
import keras_cv
import tensorflow as tf

from .mod_attendance import (
    ConstantScale,
    FusionWithGamma,
    GateEntropyLoss,
    LearnableAlpha,
    ScalarMultiply,
)


@dataclass(frozen=True)
class ModelSpec:
    """What every family needs and no family chooses: input shapes and training protocol."""

    image_size: tuple
    preset: str
    encoded_dim: int
    optimizer: Any      # template: only its config is read, the instance is never shared
    loss: Any


def build_vanilla(setting, spec, run_eagerly=False):
    """Concatenation baseline: no explicit modality adjustment."""
    image_input = keras.Input(shape=(*spec.image_size, 3), name="images")
    feat_input = keras.Input(shape=(spec.encoded_dim,), name="features")
    inp = {"images": image_input, "features": feat_input}

    backbone = keras_cv.models.EfficientNetV2Backbone.from_preset(spec.preset)
    x1 = backbone(image_input)
    x1 = keras.layers.GlobalAveragePooling2D()(x1)
    x1 = keras.layers.Dropout(0.2)(x1)

    x2 = keras.layers.Dense(96, activation="selu")(feat_input)
    x2 = keras.layers.Dense(128, activation="selu")(x2)
    x2 = keras.layers.Dropout(0.1)(x2)

    concat = keras.layers.Concatenate()([x1, x2])
    out = keras.layers.Dense(1, activation="sigmoid", dtype="float32")(concat)

    model = keras.Model(inp, out, name="vanilla")

    opt = keras.optimizers.Adam.from_config(spec.optimizer.get_config())
    model.compile(
        optimizer=opt,
        loss=spec.loss,
        metrics=[keras.metrics.AUC(name="auc")],
        run_eagerly=run_eagerly,
        jit_compile=not run_eagerly,
    )
    return model


def build_fixed_scaling(setting, spec, run_eagerly=False):
    """Fixed modality scaling: alpha on the tabular branch, 1 - alpha on the image branch."""
    alpha = float(setting["alpha"])
    beta = 1.0 - alpha

    image_input = keras.Input(shape=(*spec.image_size, 3), name="images")
    feat_input = keras.Input(shape=(spec.encoded_dim,), name="features")
    inp = {"images": image_input, "features": feat_input}

    backbone = keras_cv.models.EfficientNetV2Backbone.from_preset(spec.preset)
    x1 = backbone(image_input)
    x1 = keras.layers.GlobalAveragePooling2D()(x1)
    x1 = keras.layers.Dropout(0.2)(x1)

    x2 = keras.layers.Dense(96, activation="selu")(feat_input)
    x2 = keras.layers.Dense(128, activation="selu")(x2)
    x2 = keras.layers.Dropout(0.1)(x2)
    x1_weighted = ConstantScale(beta, name="image_weighting")(x1)
    x2_weighted = ConstantScale(alpha, name="tabular_weighting")(x2)

    concat = keras.layers.Concatenate(name="concat")([x1_weighted, x2_weighted])
    out = keras.layers.Dense(1, activation="sigmoid", dtype="float32")(concat)

    model = keras.Model(inp, out, name="fixed_scaling")
    opt = keras.optimizers.Adam.from_config(spec.optimizer.get_config())
    model.compile(
        optimizer=opt,
        loss=spec.loss,
        metrics=[keras.metrics.AUC(name="auc")],
        run_eagerly=run_eagerly,
        jit_compile=not run_eagerly,
    )
    return model


def build_learnable_scaling(setting, spec, run_eagerly=False):
    """Learnable global scaling: one sigmoid alpha, shared by every sample."""
    init_alpha = float(setting.get("init_alpha", 0.5))

    image_input = keras.Input(shape=(*spec.image_size, 3), name="images")
    feat_input = keras.Input(shape=(spec.encoded_dim,), name="features")
    inp = {"images": image_input, "features": feat_input}

    backbone = keras_cv.models.EfficientNetV2Backbone.from_preset(spec.preset)
    x1 = backbone(image_input)
    x1 = keras.layers.GlobalAveragePooling2D()(x1)
    x1 = keras.layers.Dropout(0.2)(x1)

    x2 = keras.layers.Dense(96, activation="selu")(feat_input)
    x2 = keras.layers.Dense(128, activation="selu")(x2)
    x2 = keras.layers.Dropout(0.1)(x2)

    alpha, beta = LearnableAlpha(init_alpha=init_alpha, name="learnable_alpha")(x1)
    x1_weighted = ScalarMultiply(name="image_weighting")([x1, beta])
    x2_weighted = ScalarMultiply(name="tabular_weighting")([x2, alpha])

    concat = keras.layers.Concatenate(name="concat")([x1_weighted, x2_weighted])
    out = keras.layers.Dense(1, activation="sigmoid", dtype="float32")(concat)

    model = keras.Model(inp, out, name="learnable_scaling")

    opt = keras.optimizers.Adam.from_config(spec.optimizer.get_config())
    model.compile(
        optimizer=opt,
        loss=spec.loss,
        metrics=[keras.metrics.AUC(name="auc")],
        run_eagerly=run_eagerly,
        jit_compile=not run_eagerly,
    )
    return model


def build_loss_regularized(setting, spec, run_eagerly=False):
    """Vanilla architecture plus a modality penalty in the LOSS: same layers, same widths."""
    lambda_reg = float(setting.get("lambda_reg", 1e-4))
    penalty_on = setting.get("penalty_on", "tabular")            # "tabular" | "image" | "both"
    reg_mode = setting.get("reg_mode", "per_sample")             # "per_sample" | "per_batch"
    lambda_sym = float(setting.get("lambda_sym", 0.0))           # symmetry strength
    sym_target_tab = float(setting.get("sym_target_tab", 0.5))   # 0.5 => equal influence

    image_input = keras.Input(shape=(*spec.image_size, 3), name="images")
    feat_input = keras.Input(shape=(spec.encoded_dim,), name="features")
    inp = {"images": image_input, "features": feat_input}

    backbone = keras_cv.models.EfficientNetV2Backbone.from_preset(spec.preset)
    x1 = backbone(image_input)
    x1 = keras.layers.GlobalAveragePooling2D()(x1)
    x1 = keras.layers.Dropout(0.2)(x1)

    x2 = keras.layers.Dense(96, activation="selu")(feat_input)
    x2 = keras.layers.Dense(128, activation="selu")(x2)
    x2 = keras.layers.Dropout(0.1)(x2)

    concat = keras.layers.Concatenate(name="concat")([x1, x2])
    out_layer = keras.layers.Dense(1, activation="sigmoid", dtype="float32", name="output")
    out = out_layer(concat)
    model = keras.Model(inp, out, name="loss_regularization")

    dim_img = int(x1.shape[-1])
    dim_tab = int(x2.shape[-1])
    smoothing = float(spec.loss.label_smoothing)

    def _modality_norms():
        W = out_layer.kernel  # [dim_img + dim_tab, 1]
        W_img = W[:dim_img]
        W_tab = W[dim_img:dim_img + dim_tab]
        # mean square (not sum) avoids bias from different branch dims
        n_img = tf.reduce_mean(tf.square(W_img))
        n_tab = tf.reduce_mean(tf.square(W_tab))
        return n_img, n_tab

    def _reg_term():
        n_img, n_tab = _modality_norms()

        if penalty_on == "tabular":
            mag_term = lambda_reg * n_tab
        elif penalty_on == "image":
            mag_term = lambda_reg * n_img
        elif penalty_on == "both":
            mag_term = lambda_reg * (n_img + n_tab)
        else:
            raise ValueError(f"Unsupported penalty_on={penalty_on}")

        # soft symmetry: push tab share toward target (0.5 = symmetric)
        tab_share = n_tab / (n_img + n_tab + 1e-8)
        sym_term = lambda_sym * tf.square(tab_share - sym_target_tab)

        return mag_term + sym_term

    def weighted_loss_ps(y_true, y_pred):
        base = keras.losses.binary_crossentropy(y_true, y_pred, label_smoothing=smoothing)
        return base + _reg_term()

    # per_batch reduces to a scalar, which silently disables class_weight; run_experiment raises on it.
    def weighted_loss_pb(y_true, y_pred):
        base = keras.losses.binary_crossentropy(y_true, y_pred, label_smoothing=smoothing)
        base = tf.reduce_mean(base)
        return base + _reg_term()

    loss_fn = weighted_loss_ps if reg_mode == "per_sample" else weighted_loss_pb

    opt = keras.optimizers.Adam.from_config(spec.optimizer.get_config())
    model.compile(
        optimizer=opt,
        loss=loss_fn,
        metrics=[keras.metrics.AUC(name="auc")],
        run_eagerly=run_eagerly,
        jit_compile=not run_eagerly,
    )
    return model


def build_gamma_fusion(setting, spec, run_eagerly=False):
    """Controlled fusion: both branches projected to one width, then added under gammas."""
    gamma_img = float(setting.get("gamma_img", 1.0))
    gamma_tab = float(setting.get("gamma_tab", 1.0))

    image_input = keras.Input(shape=(*spec.image_size, 3), name="images")
    feat_input = keras.Input(shape=(spec.encoded_dim,), name="features")
    inp = {"images": image_input, "features": feat_input}

    backbone = keras_cv.models.EfficientNetV2Backbone.from_preset(spec.preset)
    x1 = backbone(image_input)
    x1 = keras.layers.GlobalAveragePooling2D()(x1)
    x1 = keras.layers.Dropout(0.2)(x1)
    x1 = keras.layers.Dense(128, activation="linear", name="img_proj")(x1)

    x2 = keras.layers.Dense(96, activation="selu")(feat_input)
    x2 = keras.layers.Dense(128, activation="selu")(x2)
    x2 = keras.layers.Dropout(0.1)(x2)

    fused = FusionWithGamma(gamma_img=gamma_img, gamma_tab=gamma_tab,
                            name="fusion_with_gamma")([x1, x2])
    out = keras.layers.Dense(1, activation="sigmoid", dtype="float32")(fused)

    model = keras.Model(inp, out, name="controlled_fusion")
    opt = keras.optimizers.Adam.from_config(spec.optimizer.get_config())
    model.compile(
        optimizer=opt,
        loss=spec.loss,
        metrics=[keras.metrics.AUC(name="auc")],
        run_eagerly=run_eagerly,
        jit_compile=not run_eagerly,
    )
    return model


def build_adaptive_gating(setting, spec, run_eagerly=False):
    """Adaptive gating: a per-sample softmax over the two branches, penalised by its entropy."""
    entropy_weight = float(setting.get("entropy_weight", 1e-4))

    image_input = keras.Input(shape=(*spec.image_size, 3), name="images")
    feat_input = keras.Input(shape=(spec.encoded_dim,), name="features")
    inp = {"images": image_input, "features": feat_input}

    backbone = keras_cv.models.EfficientNetV2Backbone.from_preset(spec.preset)
    x1 = backbone(image_input)
    x1 = keras.layers.GlobalAveragePooling2D()(x1)
    x1 = keras.layers.Dropout(0.2)(x1)
    x1 = keras.layers.Dense(256, activation="selu", name="img_emb")(x1)

    x2 = keras.layers.Dense(96, activation="selu")(feat_input)
    x2 = keras.layers.Dense(128, activation="selu")(x2)
    x2 = keras.layers.Dropout(0.1)(x2)
    x2 = keras.layers.Dense(256, activation="selu", name="tab_emb")(x2)

    fusion_input = keras.layers.Concatenate(name="fusion_input")([x1, x2])
    gate = keras.layers.Dense(64, activation="relu")(fusion_input)
    gate = keras.layers.Dense(2, activation="softmax", name="gate_weights")(gate)
    gate = GateEntropyLoss(weight=entropy_weight, name="gate_entropy_loss")(gate)

    w_img = keras.layers.Lambda(lambda w: w[:, 0:1], name="w_img")(gate)
    w_tab = keras.layers.Lambda(lambda w: w[:, 1:2], name="w_tab")(gate)

    x1_w = keras.layers.Multiply(name="weighted_img")([x1, w_img])
    x2_w = keras.layers.Multiply(name="weighted_tab")([x2, w_tab])
    fused = keras.layers.Add(name="fused")([x1_w, x2_w])

    out = keras.layers.Dense(1, activation="sigmoid", dtype="float32", name="output")(fused)

    model = keras.Model(inp, out, name="adaptive_gating")
    opt = keras.optimizers.Adam.from_config(spec.optimizer.get_config())
    model.compile(
        optimizer=opt,
        loss=spec.loss,
        metrics=[keras.metrics.AUC(name="auc")],
        run_eagerly=run_eagerly,
        jit_compile=not run_eagerly,
    )
    return model


# Keyed by impl_name: the same string the sweeps pass to run_experiment and the runs are
# filed under. Every builder takes (setting, spec, run_eagerly), the signature the runner's
# `build_model_from_setting_fn` already expects.
FAMILIES = {
    "vanilla": build_vanilla,
    "fixed_scaling": build_fixed_scaling,
    "learnable_scaling": build_learnable_scaling,
    "loss_regularization": build_loss_regularized,
    "controlled_fusion": build_gamma_fusion,
    "adaptive_gating": build_adaptive_gating,
}