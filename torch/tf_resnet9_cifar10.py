import csv
import datetime
import os
import time
import numpy as np

os.environ["TF_XLA_FLAGS"] = "--tf_xla_auto_jit=2"

import tensorflow as tf
from dotenv import load_dotenv
load_dotenv()

# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

def load_cifar10_bin(root: str, train: bool):
    """Load CIFAR-10 binary-format files into (images, labels) numpy arrays."""
    if train:
        batch_files = [f"data_batch_{i}.bin" for i in range(1, 6)]
    else:
        batch_files = ["test_batch.bin"]

    all_images = []
    all_labels = []

    for fname in batch_files:
        path = os.path.join(root, fname)
        if not os.path.isfile(path):
            raise FileNotFoundError(f"File not found: {path}")
        with open(path, "rb") as f:
            arr = np.frombuffer(f.read(), dtype=np.uint8)
            arr = arr.reshape(-1, 3073)
            labels = arr[:, 0]
            images = arr[:, 1:].reshape(-1, 3, 32, 32)  # NCHW
            all_images.append(images)
            all_labels.append(labels)

    images = np.concatenate(all_images, axis=0).astype(np.float32) / 255.0
    labels = np.concatenate(all_labels, axis=0).astype(np.int32)

    # NCHW -> NHWC for TensorFlow
    images = images.transpose(0, 2, 3, 1)
    return images, labels


CIFAR10_MEAN = np.array([0.49139968, 0.48215827, 0.44653124], dtype=np.float32)
CIFAR10_STD  = np.array([0.24703233, 0.24348505, 0.26158768], dtype=np.float32)


def normalize(img):
    return (img - CIFAR10_MEAN) / CIFAR10_STD


def make_dataset(images, labels, batch_size: int, shuffle: bool) -> tf.data.Dataset:
    ds = tf.data.Dataset.from_tensor_slices((images, labels))
    if shuffle:
        ds = ds.shuffle(buffer_size=len(images), reshuffle_each_iteration=True)
    ds = ds.map(
        lambda x, y: (normalize(x), y),
        num_parallel_calls=tf.data.AUTOTUNE,
    )
    ds = ds.batch(batch_size, drop_remainder=False)
    ds = ds.prefetch(tf.data.AUTOTUNE)
    return ds


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

class BasicResidualBlock(tf.keras.layers.Layer):
    def __init__(self, channels: int, **kwargs):
        super().__init__(**kwargs)
        reg = tf.keras.regularizers.l2(3e-4)
        self.conv1 = tf.keras.layers.Conv2D(
            channels, 3, strides=1, padding="same", use_bias=True,
            kernel_regularizer=reg)
        self.bn1   = tf.keras.layers.BatchNormalization(
            momentum=0.9, epsilon=1e-5)
        self.conv2 = tf.keras.layers.Conv2D(
            channels, 3, strides=1, padding="same", use_bias=True,
            kernel_regularizer=reg)
        self.bn2   = tf.keras.layers.BatchNormalization(
            momentum=0.9, epsilon=1e-5)

    def call(self, x, training=False):
        identity = x
        out = tf.nn.relu(self.bn1(self.conv1(x), training=training))
        out = self.bn2(self.conv2(out), training=training)
        return tf.nn.relu(out + identity)


def build_resnet9(num_classes: int = 10) -> tf.keras.Model:
    inp = tf.keras.Input(shape=(32, 32, 3))

    reg = tf.keras.regularizers.l2(3e-4)

    # Block 1
    x = tf.keras.layers.Conv2D(64, 3, strides=1, padding="same",
                                use_bias=True, kernel_regularizer=reg)(inp)
    x = tf.keras.layers.BatchNormalization(momentum=0.9, epsilon=1e-5)(x)
    x = tf.keras.layers.ReLU()(x)

    # Block 2
    x = tf.keras.layers.Conv2D(128, 3, strides=1, padding="same",
                                use_bias=True, kernel_regularizer=reg)(x)
    x = tf.keras.layers.BatchNormalization(momentum=0.9, epsilon=1e-5)(x)
    x = tf.keras.layers.ReLU()(x)
    x = tf.keras.layers.MaxPool2D(pool_size=2, strides=2, padding="valid")(x)

    # Residual blocks 1 & 2
    x = BasicResidualBlock(128)(x)
    x = BasicResidualBlock(128)(x)

    # Block 3
    x = tf.keras.layers.Conv2D(256, 3, strides=1, padding="same",
                                use_bias=True, kernel_regularizer=reg)(x)
    x = tf.keras.layers.BatchNormalization(momentum=0.9, epsilon=1e-5)(x)
    x = tf.keras.layers.ReLU()(x)
    x = tf.keras.layers.MaxPool2D(pool_size=2, strides=2, padding="valid")(x)

    # Residual blocks 3 & 4
    x = BasicResidualBlock(256)(x)
    x = BasicResidualBlock(256)(x)

    # Block 4
    x = tf.keras.layers.Conv2D(512, 3, strides=1, padding="same",
                                use_bias=True, kernel_regularizer=reg)(x)
    x = tf.keras.layers.BatchNormalization(momentum=0.9, epsilon=1e-5)(x)
    x = tf.keras.layers.ReLU()(x)
    x = tf.keras.layers.MaxPool2D(pool_size=2, strides=2, padding="valid")(x)

    # Residual block 5
    x = BasicResidualBlock(512)(x)

    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    out = tf.keras.layers.Dense(num_classes, use_bias=True,
                                 kernel_regularizer=reg)(x)

    return tf.keras.Model(inputs=inp, outputs=out)


# ---------------------------------------------------------------------------
# Training step (XLA-compiled)
# ---------------------------------------------------------------------------

@tf.function(jit_compile=True)
def train_step(model, optimizer, loss_fn, images, labels):
    with tf.GradientTape() as tape:
        logits = model(images, training=True)
        loss   = loss_fn(labels, logits)
        loss  += tf.add_n(model.losses) if model.losses else 0.0
    grads = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(grads, model.trainable_variables))
    preds   = tf.argmax(logits, axis=1, output_type=tf.int32)
    correct = tf.reduce_sum(tf.cast(tf.equal(preds, labels), tf.int32))
    return loss, correct


@tf.function(jit_compile=True)
def val_step(model, loss_fn, images, labels):
    logits  = model(images, training=False)
    loss    = loss_fn(labels, logits)
    preds   = tf.argmax(logits, axis=1, output_type=tf.int32)
    correct = tf.reduce_sum(tf.cast(tf.equal(preds, labels), tf.int32))
    return loss, correct


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    gpus = tf.config.list_physical_devices("GPU")
    if gpus:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        print(f">>> Running on GPU(s): {[g.name for g in gpus]}")
    else:
        print(">>> No GPU found, running on CPU")

    epochs     = int(os.getenv("EPOCHS",     "10"))
    batch_size = int(os.getenv("BATCH_SIZE", "128"))
    lr_initial = float(os.getenv("LR_INITIAL", "0.001"))
    data_root  = os.getenv("CIFAR10_BIN_ROOT", "data/cifar-10-batches-bin")

    print(f">>> Using CIFAR-10 bin data at: {data_root}")
    print(f">>> Epochs: {epochs}, Batch size: {batch_size}, LR: {lr_initial}")

    train_images, train_labels = load_cifar10_bin(data_root, train=True)
    test_images,  test_labels  = load_cifar10_bin(data_root, train=False)

    train_ds = make_dataset(train_images, train_labels, batch_size, shuffle=True)
    test_ds  = make_dataset(test_images,  test_labels,  batch_size, shuffle=False)

    model = build_resnet9(num_classes=10)

    loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)

    # StepLR: multiply by gamma every step_size epochs
    step_size = 5
    gamma     = 0.1

    def lr_schedule(epoch_0indexed):
        """Returns the LR for a given 0-indexed epoch."""
        return lr_initial * (gamma ** (epoch_0indexed // step_size))

    optimizer = tf.keras.optimizers.Adam(
        learning_rate=lr_initial,
        beta_1=0.9,
        beta_2=0.999,
        epsilon=1e-3,
    )

    # Logging
    ts      = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    log_dir = "logs"
    os.makedirs(log_dir, exist_ok=True)

    batch_csv_path = os.path.join(log_dir, f"tf_cifar10_resnet9_batch_{ts}.csv")
    epoch_csv_path = os.path.join(log_dir, f"tf_cifar10_resnet9_epoch_{ts}.csv")
    val_csv_path   = os.path.join(log_dir, f"tf_cifar10_resnet9_val_{ts}.csv")

    batch_csv_file = open(batch_csv_path, "w", newline="")
    epoch_csv_file = open(epoch_csv_path, "w", newline="")
    val_csv_file   = open(val_csv_path,   "w", newline="")

    batch_writer = csv.writer(batch_csv_file)
    epoch_writer = csv.writer(epoch_csv_file)
    val_writer   = csv.writer(val_csv_file)

    batch_writer.writerow(["epoch", "step", "loss", "accuracy_pct", "time_ms"])
    epoch_writer.writerow(["epoch", "train_loss", "train_accuracy_pct",
                            "val_loss", "val_accuracy_pct"])
    val_writer.writerow(["epoch", "step", "loss", "accuracy_pct"])

    n_train = len(train_images)
    n_test  = len(test_images)

    for epoch in range(1, epochs + 1):
        print(f"\n===== Epoch {epoch}/{epochs} =====")

        # Update LR per StepLR schedule
        new_lr = lr_schedule(epoch - 1)
        optimizer.learning_rate.assign(new_lr)

        epoch_start    = time.time()
        running_loss   = 0.0
        running_correct = 0
        running_total   = 0

        for batch_idx, (images, labels) in enumerate(train_ds):
            step_start = time.time()
            loss, correct = train_step(model, optimizer, loss_fn, images, labels)
            step_ms = int((time.time() - step_start) * 1000)

            batch_n      = images.shape[0]
            batch_loss   = float(loss.numpy())
            batch_acc    = 100.0 * int(correct.numpy()) / batch_n

            running_loss    += batch_loss * batch_n
            running_correct += int(correct.numpy())
            running_total   += batch_n

            batch_writer.writerow(
                [epoch, batch_idx + 1, f"{batch_loss:.6f}", f"{batch_acc:.4f}", step_ms]
            )
            batch_csv_file.flush()

            if (batch_idx + 1) % 100 == 0:
                print(
                    f"[Train Batch {batch_idx+1}] "
                    f"Loss: {batch_loss:.4f} | Acc: {batch_acc:.2f}% | "
                    f"Step time: {step_ms}ms"
                )

        train_loss = running_loss / running_total
        train_acc  = 100.0 * running_correct / running_total

        # Validation
        val_loss_sum   = 0.0
        val_correct    = 0
        val_total      = 0

        for val_step_idx, (images, labels) in enumerate(test_ds):
            loss, correct = val_step(model, loss_fn, images, labels)

            batch_n    = images.shape[0]
            step_loss  = float(loss.numpy())
            step_acc   = 100.0 * int(correct.numpy()) / batch_n

            val_loss_sum += step_loss * batch_n
            val_correct  += int(correct.numpy())
            val_total    += batch_n

            val_writer.writerow(
                [epoch, val_step_idx + 1, f"{step_loss:.6f}", f"{step_acc:.4f}"]
            )
        val_csv_file.flush()

        val_loss = val_loss_sum / val_total
        val_acc  = 100.0 * val_correct / val_total
        epoch_time = time.time() - epoch_start

        epoch_writer.writerow(
            [epoch, f"{train_loss:.6f}", f"{train_acc:.4f}",
             f"{val_loss:.6f}", f"{val_acc:.4f}"]
        )
        epoch_csv_file.flush()

        print(
            f"Epoch {epoch}/{epochs} Completed in {epoch_time:.2f}s\n"
            f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}%\n"
            f"Val   Loss: {val_loss:.4f} | Val   Acc: {val_acc:.2f}%"
        )

    batch_csv_file.close()
    epoch_csv_file.close()
    val_csv_file.close()

    print(f"\n>>> Logs saved to {log_dir}/tf_cifar10_resnet9_*_{ts}.csv")
    print("\n>>> CIFAR-10 ResNet-9 (TensorFlow + XLA) training completed.")


if __name__ == "__main__":
    main()
