import tensorflow as tf

def preprocess(image, label):
    image = tf.cast(image, tf.float32) / 255.
    return image, label

augment_layer = tf.keras.Sequential([
tf.keras.layers.RandomFlip("horizontal"),
tf.keras.layers.RandomTranslation(.1, .1),
])

def augment(image, label):
    return augment_layer(image, training=True), label

def make_datasets(dataset, training=True, batch_size=32):
    dataset = dataset.map(preprocess, num_parallel_calls=tf.data.AUTOTUNE)
    if training:
        dataset = dataset.cache()
        dataset = dataset.shuffle(1000)
        dataset = dataset.map(augment, num_parallel_calls=tf.data.AUTOTUNE)
    return dataset.batch(batch_size).prefetch(tf.data.AUTOTUNE)