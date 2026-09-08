import tensorflow as tf
import keras_cv

# An image store declares its own encoding; the ones written before it did hold JPEG bytes.
_DECODERS = {"jpeg_bytes": tf.io.decode_jpeg, "png_bytes": tf.io.decode_png}


def build_augmenter():
    aug_layers = [
        keras_cv.layers.RandomCutout(height_factor=(0.02, 0.06), width_factor=(0.02, 0.06)),
        keras_cv.layers.RandomFlip(mode="horizontal"),
    ]
    aug_layers = [keras_cv.layers.RandomApply(x, rate=0.5) for x in aug_layers]
    augmenter = keras_cv.layers.Augmenter(aug_layers)

    def augment(inp, label):
        aug_data = {"images": inp["images"]}
        aug_data = augmenter(aug_data)
        inp["images"] = aug_data["images"]
        return inp, label

    return augment


def build_decoder(hdf5, with_labels: bool, target_size, num_classes: int):
    decode = _DECODERS[hdf5.attrs.get("format", "jpeg_bytes")]

    def decode_image(inp):
        image = decode(inp["images"], channels=3)
        image = tf.image.resize(image, size=target_size, method="area")
        image = tf.cast(image, tf.float32) / 255.0
        inp["images"] = tf.reshape(image, [*target_size, 3])
        return inp

    def decode_label(label):
        label = tf.cast(label, tf.float32)
        return tf.reshape(label, [num_classes])

    def decode_with_labels(inp, label):
        inp = decode_image(inp)
        return inp, decode_label(label)

    return decode_with_labels if with_labels else decode_image


def build_dataset(
    row_ids,
    hdf5,
    features,
    labels=None,
    batch_size=32,
    image_size=(256, 256),
    num_classes=1,
    seed=42,
    decode_fn=None,
    augment_fn=None,
    augment=False,
    shuffle=1024,
    cache=True,
    drop_remainder=False,
):
    if decode_fn is None:
        decode_fn = build_decoder(hdf5, with_labels=(labels is not None), target_size=image_size,
                                  num_classes=num_classes)
    if augment_fn is None:
        augment_fn = build_augmenter()

    AUTO = tf.data.AUTOTUNE

    # NOTE: This keeps your current behavior (loads all images into RAM).
    # Only safe if len(row_ids) is small or your machine has a lot of RAM.
    images = [hdf5[i][()] for i in row_ids]

    inp = {"images": images, "features": features}
    slices = (inp, labels) if labels is not None else inp

    ds = tf.data.Dataset.from_tensor_slices(slices)
    ds = ds.cache() if cache else ds
    ds = ds.map(decode_fn, num_parallel_calls=AUTO)

    if shuffle:
        ds = ds.shuffle(shuffle, seed=seed)
        opt = tf.data.Options()
        opt.deterministic = False
        ds = ds.with_options(opt)

    ds = ds.batch(batch_size, drop_remainder=drop_remainder)
    ds = ds.map(augment_fn, num_parallel_calls=AUTO) if augment else ds
    ds = ds.prefetch(AUTO)
    return ds