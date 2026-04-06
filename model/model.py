import tensorflow as tf
import numpy as np
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.optimizers import Adam

# Image size
IMG_H, IMG_W = 466, 405
BATCH_SIZE = 16

# ─────────────────────────────
# Load dataset (images + labels)
# ─────────────────────────────
def load_dataset(image_paths, angles, shuffle=True):

    def process(path, angle):
        img = tf.io.read_file(path)
        img = tf.image.decode_png(img, channels=3)
        img = tf.cast(img, tf.float32) / 255.0
        img = tf.image.resize(img, (IMG_H, IMG_W))
        return img, angle

    ds = tf.data.Dataset.from_tensor_slices((image_paths, angles))

    if shuffle:
        ds = ds.shuffle(1000)

    ds = ds.map(process)
    ds = ds.batch(BATCH_SIZE)
    return ds


# ─────────────────────────────
# Model
# ─────────────────────────────
def build_model():
    model = keras.Sequential([
        layers.Cropping2D(((100, 0), (0, 0)), input_shape=(IMG_H, IMG_W, 3)),
        layers.Resizing(66, 200),

        layers.Conv2D(24, 5, strides=2, activation='relu'),
        layers.Conv2D(36, 5, strides=2, activation='relu'),
        layers.Conv2D(48, 5, strides=2, activation='relu'),
        layers.Conv2D(64, 3, activation='relu'),
        layers.Conv2D(64, 3, activation='relu'),

        layers.Flatten(),
        layers.Dense(100, activation='relu'),
        layers.Dense(50, activation='relu'),
        layers.Dense(10, activation='relu'),
        layers.Dense(1)  # steering output
    ])

    model.compile(
        optimizer=Adam(1e-4),
        loss='mse',
        metrics=['mae']
    )

    return model


# ─────────────────────────────
# Train
# ─────────────────────────────
if __name__ == "__main__":

    # Example (replace with your real data)
    image_paths = ["data/img1.png", "data/img2.png"]
    angles = np.array([0.1, -0.2], dtype=np.float32)

    train_ds = load_dataset(image_paths, angles)

    model = build_model()
    model.summary()

    model.fit(train_ds, epochs=10)

    # Save model
    model.save("model/model.keras")
    print("Model saved")