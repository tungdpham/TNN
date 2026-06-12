#!/usr/bin/env python3
"""
Unified single-GPU trainer for SYNET TensorFlow reference models.

Usage:
    python tensorflow_trainer.py --model resnet9_cifar10
    python tensorflow_trainer.py --model wrn16_8_cifar100 --epochs 50
    python tensorflow_trainer.py --model resnet50_tiny_imagenet --batch-size 128
    python tensorflow_trainer.py --model resnet50_imagenet100 --batch-size 64
    python tensorflow_trainer.py --model gpt2_small --batch-size 8

Environment variables can override defaults (see .env or shell):
    EPOCHS, BATCH_SIZE, LR_INITIAL, CIFAR10_BIN_ROOT, CIFAR100_BIN_ROOT,
    TINY_IMAGENET_ROOT, IMAGENET100_ROOT, OPENWEBTEXT_PATH, etc.
"""
import argparse
import csv
import datetime
import math
import os
import time
from pathlib import Path
from typing import Dict, Any, Tuple

import numpy as np
import tensorflow as tf
from tensorflow.keras import layers
from dotenv import load_dotenv

load_dotenv()


# ======================== Normalization constants ========================

CIFAR10_MEAN  = np.array([0.49139968, 0.48215827, 0.44653124], dtype=np.float32)
CIFAR10_STD   = np.array([0.24703233, 0.24348505, 0.26158768], dtype=np.float32)

CIFAR100_MEAN = np.array([0.50707516, 0.48654887, 0.44091784], dtype=np.float32)
CIFAR100_STD  = np.array([0.26733429, 0.25643846, 0.27615047], dtype=np.float32)

TINY_MEAN = [0.4802, 0.4481, 0.3975]
TINY_STD  = [0.2770, 0.2691, 0.2821]

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]

# BatchNorm: PyTorch momentum=0.1 (update rate) <-> TF momentum=0.9 (decay rate)
BN_MOMENTUM = 0.9
BN_EPS      = 1e-5


# ======================== Datasets ========================

def make_cifar10_dataset(root: str, train: bool, batch_size: int,
                          num_workers: int) -> Tuple[tf.data.Dataset, int]:
    """Load CIFAR-10 binary format, return (tf.data.Dataset, num_samples)."""
    batch_files = (
        [f"data_batch_{i}.bin" for i in range(1, 6)] if train else ["test_batch.bin"]
    )
    all_images, all_labels = [], []
    for fname in batch_files:
        path = os.path.join(root, fname)
        if not os.path.isfile(path):
            raise FileNotFoundError(f"File not found: {path}")
        with open(path, "rb") as f:
            arr = np.frombuffer(f.read(), dtype=np.uint8).reshape(-1, 3073)
        all_labels.append(arr[:, 0])
        all_images.append(arr[:, 1:].reshape(-1, 3, 32, 32))

    images = np.concatenate(all_images, axis=0)          # (N, 3, 32, 32) NCHW
    labels = np.concatenate(all_labels, axis=0).astype(np.int64)

    # NCHW -> NHWC, normalize
    images = np.transpose(images, (0, 2, 3, 1)).astype(np.float32) / 255.0
    images = (images - CIFAR10_MEAN) / CIFAR10_STD

    def gen():
        for img, lbl in zip(images, labels):
            yield img, lbl

    ds = tf.data.Dataset.from_generator(
        gen,
        output_signature=(
            tf.TensorSpec(shape=(32, 32, 3), dtype=tf.float32),
            tf.TensorSpec(shape=(),          dtype=tf.int64),
        ),
    )
    if train:
        ds = ds.shuffle(buffer_size=1000, reshuffle_each_iteration=True)
    ds = ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)
    return ds, int(len(labels))


def make_cifar100_dataset(root: str, train: bool, batch_size: int,
                           num_workers: int) -> Tuple[tf.data.Dataset, int]:
    """Load CIFAR-100 binary format, return (tf.data.Dataset, num_samples)."""
    fname = "train.bin" if train else "test.bin"
    path = os.path.join(root, fname)
    if not os.path.isfile(path):
        raise FileNotFoundError(f"File not found: {path}")
    with open(path, "rb") as f:
        arr = np.frombuffer(f.read(), dtype=np.uint8).reshape(-1, 3074)

    labels = arr[:, 1].astype(np.int64)          # fine labels
    images = arr[:, 2:].reshape(-1, 3, 32, 32)

    # NCHW -> NHWC, normalize
    images = np.transpose(images, (0, 2, 3, 1)).astype(np.float32) / 255.0
    images = (images - CIFAR100_MEAN) / CIFAR100_STD

    def gen():
        for img, lbl in zip(images, labels):
            yield img, lbl

    ds = tf.data.Dataset.from_generator(
        gen,
        output_signature=(
            tf.TensorSpec(shape=(32, 32, 3), dtype=tf.float32),
            tf.TensorSpec(shape=(),          dtype=tf.int64),
        ),
    )
    if train:
        ds = ds.shuffle(buffer_size=1000, reshuffle_each_iteration=True)
    ds = ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)
    return ds, int(len(labels))


def make_tiny_imagenet_dataset(root: str, train: bool, batch_size: int,
                                num_workers: int) -> Tuple[tf.data.Dataset, int]:
    """Tiny ImageNet dataset, return (tf.data.Dataset, num_samples)."""
    root_p = Path(root)
    train_dir = root_p / "train"
    classes = sorted(d.name for d in train_dir.iterdir() if d.is_dir())
    class_to_idx = {cls: idx for idx, cls in enumerate(classes)}

    samples = []
    if train:
        for cls in classes:
            cls_dir = train_dir / cls / "images"
            if not cls_dir.exists():
                continue
            for p in cls_dir.glob("*.JPEG"):
                samples.append((str(p), class_to_idx[cls]))
    else:
        val_dir = root_p / "val"
        annotations_file = val_dir / "val_annotations.txt"
        with open(annotations_file, "r") as f:
            for line in f:
                parts = line.strip().split("\t")
                img_path = val_dir / "images" / parts[0]
                cls = parts[1]
                if cls in class_to_idx and img_path.exists():
                    samples.append((str(img_path), class_to_idx[cls]))

    paths  = [s[0] for s in samples]
    labels = [s[1] for s in samples]

    mean_t = tf.constant(TINY_MEAN, dtype=tf.float32)
    std_t  = tf.constant(TINY_STD,  dtype=tf.float32)

    do_aug = train

    def load_image(path, label):
        raw = tf.io.read_file(path)
        img = tf.image.decode_jpeg(raw, channels=3)
        img = tf.image.resize(img, [64, 64])
        img = tf.cast(img, tf.float32) / 255.0
        if do_aug:
            img = tf.image.random_flip_left_right(img)
        img = (img - mean_t) / std_t
        return img, label

    ds = tf.data.Dataset.from_tensor_slices((paths, labels))
    if train:
        ds = ds.shuffle(len(samples), reshuffle_each_iteration=True)
    ds = ds.map(load_image, num_parallel_calls=tf.data.AUTOTUNE)
    ds = ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)
    return ds, len(samples)


def make_imagenet100_dataset(root: str, train: bool, batch_size: int,
                              num_workers: int) -> Tuple[tf.data.Dataset, int]:
    """ImageNet-100 dataset, return (tf.data.Dataset, num_samples)."""
    root_p = Path(root)

    all_classes: set = set()
    for sub in ["train.X1", "train.X2", "train.X3", "train.X4"]:
        d = root_p / sub
        if d.exists():
            all_classes.update(x.name for x in d.iterdir() if x.is_dir())
    classes = sorted(all_classes)
    class_to_idx = {cls: idx for idx, cls in enumerate(classes)}

    samples = []
    if train:
        for sub in ["train.X1", "train.X2", "train.X3", "train.X4"]:
            data_dir = root_p / sub
            if not data_dir.exists():
                continue
            for cls in classes:
                img_dir = data_dir / cls
                if not img_dir.exists():
                    continue
                for ext in ["*.JPEG", "*.jpg", "*.png"]:
                    for p in img_dir.glob(ext):
                        samples.append((str(p), class_to_idx[cls]))
    else:
        data_dir = root_p / "val.X"
        for cls in classes:
            img_dir = data_dir / cls
            if not img_dir.exists():
                continue
            for ext in ["*.JPEG", "*.jpg", "*.png"]:
                for p in img_dir.glob(ext):
                    samples.append((str(p), class_to_idx[cls]))

    paths  = [s[0] for s in samples]
    labels = [s[1] for s in samples]

    mean_t = tf.constant(IMAGENET_MEAN, dtype=tf.float32)
    std_t  = tf.constant(IMAGENET_STD,  dtype=tf.float32)

    def load_train(path, label):
        raw = tf.io.read_file(path)
        img = tf.image.decode_jpeg(raw, channels=3)
        img = tf.image.resize(img, [256, 256])
        img = tf.cast(img, tf.float32) / 255.0
        img = tf.image.random_crop(img, [224, 224, 3])
        img = tf.image.random_flip_left_right(img)
        img = (img - mean_t) / std_t
        return img, label

    def load_val(path, label):
        raw = tf.io.read_file(path)
        img = tf.image.decode_jpeg(raw, channels=3)
        img = tf.image.resize(img, [256, 256])
        img = tf.cast(img, tf.float32) / 255.0
        img = tf.image.resize_with_crop_or_pad(img, 224, 224)
        img = (img - mean_t) / std_t
        return img, label

    load_fn = load_train if train else load_val

    ds = tf.data.Dataset.from_tensor_slices((paths, labels))
    if train:
        ds = ds.shuffle(len(samples), reshuffle_each_iteration=True)
    ds = ds.map(load_fn, num_parallel_calls=tf.data.AUTOTUNE)
    ds = ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)
    return ds, len(samples)


def make_openwebtext_dataset(path: str, seq_len: int, train: bool,
                              batch_size: int,
                              num_workers: int) -> Tuple[tf.data.Dataset, int]:
    """OpenWebText tokenized dataset for GPT-2, return (tf.data.Dataset, num_samples)."""
    if not os.path.isfile(path):
        raise FileNotFoundError(f"Data file not found: {path}")
    data = np.memmap(path, dtype=np.uint16, mode="r")
    n = (len(data) - 1) // seq_len

    def generator():
        for idx in range(n):
            start = idx * seq_len
            chunk = data[start : start + seq_len + 1].astype(np.int64)
            yield chunk[:-1], chunk[1:]

    ds = tf.data.Dataset.from_generator(
        generator,
        output_signature=(
            tf.TensorSpec(shape=(seq_len,), dtype=tf.int64),
            tf.TensorSpec(shape=(seq_len,), dtype=tf.int64),
        ),
    )
    if train:
        ds = ds.shuffle(1000)
    ds = ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)
    return ds, n


# ======================== Models ========================

# --- ResNet-9 for CIFAR-10 ---

class BasicResidualBlock(tf.keras.layers.Layer):
    def __init__(self, channels: int, **kwargs):
        super().__init__(**kwargs)
        self.conv1 = layers.Conv2D(channels, 3, padding="same", use_bias=True)
        self.bn1   = layers.BatchNormalization(momentum=BN_MOMENTUM, epsilon=BN_EPS)
        self.conv2 = layers.Conv2D(channels, 3, padding="same", use_bias=True)
        self.bn2   = layers.BatchNormalization(momentum=BN_MOMENTUM, epsilon=BN_EPS)

    def call(self, x, training=False):
        identity = x
        out = tf.nn.relu(self.bn1(self.conv1(x), training=training))
        out = self.bn2(self.conv2(out), training=training)
        return tf.nn.relu(out + identity)


class ResNet9CIFAR10(tf.keras.Model):
    def __init__(self, num_classes: int = 10):
        super().__init__()
        self.conv1 = layers.Conv2D(64,  3, padding="same", use_bias=True)
        self.bn1   = layers.BatchNormalization(momentum=BN_MOMENTUM, epsilon=BN_EPS)
        self.conv2 = layers.Conv2D(128, 3, padding="same", use_bias=True)
        self.bn2   = layers.BatchNormalization(momentum=BN_MOMENTUM, epsilon=BN_EPS)
        self.pool1 = layers.MaxPool2D(pool_size=2, strides=2)

        self.res1  = BasicResidualBlock(128)
        self.res2  = BasicResidualBlock(128)

        self.conv3 = layers.Conv2D(256, 3, padding="same", use_bias=True)
        self.bn3   = layers.BatchNormalization(momentum=BN_MOMENTUM, epsilon=BN_EPS)
        self.pool2 = layers.MaxPool2D(pool_size=2, strides=2)

        self.res3  = BasicResidualBlock(256)
        self.res4  = BasicResidualBlock(256)

        self.conv4 = layers.Conv2D(512, 3, padding="same", use_bias=True)
        self.bn4   = layers.BatchNormalization(momentum=BN_MOMENTUM, epsilon=BN_EPS)
        self.pool3 = layers.MaxPool2D(pool_size=2, strides=2)

        self.res5  = BasicResidualBlock(512)

        self.gavg  = layers.GlobalAveragePooling2D()
        self.fc    = layers.Dense(num_classes, use_bias=True)

    def call(self, x, training=False):
        x = tf.nn.relu(self.bn1(self.conv1(x), training=training))
        x = tf.nn.relu(self.bn2(self.conv2(x), training=training))
        x = self.pool1(x)
        x = self.res1(x, training=training)
        x = self.res2(x, training=training)
        x = tf.nn.relu(self.bn3(self.conv3(x), training=training))
        x = self.pool2(x)
        x = self.res3(x, training=training)
        x = self.res4(x, training=training)
        x = tf.nn.relu(self.bn4(self.conv4(x), training=training))
        x = self.pool3(x)
        x = self.res5(x, training=training)
        x = self.gavg(x)
        return self.fc(x)


# --- Wide ResNet-16-8 for CIFAR-100 ---

class WideResidualBlock(tf.keras.layers.Layer):
    """Pre-activation wide residual block."""
    def __init__(self, in_channels: int, out_channels: int,
                 stride: int = 1, dropout_rate: float = 0.0, **kwargs):
        super().__init__(**kwargs)
        self.bn1   = layers.BatchNormalization(momentum=BN_MOMENTUM, epsilon=BN_EPS)
        self.conv1 = layers.Conv2D(out_channels, 3, strides=stride, padding="same", use_bias=True)
        self.bn2   = layers.BatchNormalization(momentum=BN_MOMENTUM, epsilon=BN_EPS)
        self.drop  = layers.Dropout(dropout_rate) if dropout_rate > 0.0 else None
        self.conv2 = layers.Conv2D(out_channels, 3, padding="same", use_bias=True)

        self.shortcut = None
        if stride != 1 or in_channels != out_channels:
            self.shortcut = layers.Conv2D(
                out_channels, 1, strides=stride, padding="valid", use_bias=False
            )

    def call(self, x, training=False):
        sc  = self.shortcut(x) if self.shortcut is not None else x
        out = tf.nn.relu(self.bn1(x, training=training))
        out = self.conv1(out)
        out = tf.nn.relu(self.bn2(out, training=training))
        if self.drop is not None:
            out = self.drop(out, training=training)
        out = self.conv2(out)
        return out + sc


class WRN16_8CIFAR100(tf.keras.Model):
    """WRN-16-8 for CIFAR-100."""
    def __init__(self, num_classes: int = 100):
        super().__init__()
        dr       = 0.3
        c1, c2, c3 = 128, 256, 512

        self.conv1 = layers.Conv2D(16, 3, padding="same", use_bias=True)

        # Group 1: 16 -> 128, stride 1
        self.g1b1 = WideResidualBlock(16, c1, stride=1, dropout_rate=dr)
        self.g1b2 = WideResidualBlock(c1, c1, stride=1, dropout_rate=dr)

        # Group 2: 128 -> 256, stride 2 (32x32 -> 16x16)
        self.g2b1 = WideResidualBlock(c1, c2, stride=2, dropout_rate=dr)
        self.g2b2 = WideResidualBlock(c2, c2, stride=1, dropout_rate=dr)

        # Group 3: 256 -> 512, stride 2 (16x16 -> 8x8)
        self.g3b1 = WideResidualBlock(c2, c3, stride=2, dropout_rate=dr)
        self.g3b2 = WideResidualBlock(c3, c3, stride=1, dropout_rate=dr)

        # Final BN+ReLU before pooling
        self.bn_final = layers.BatchNormalization(momentum=BN_MOMENTUM, epsilon=BN_EPS)

        # Global average pool: 8x8 -> 1x1
        self.avgpool = layers.AveragePooling2D(pool_size=8, strides=1)
        self.flatten = layers.Flatten()
        self.fc      = layers.Dense(num_classes, use_bias=True)

    def call(self, x, training=False):
        x = self.conv1(x)
        x = self.g1b1(x, training=training)
        x = self.g1b2(x, training=training)
        x = self.g2b1(x, training=training)
        x = self.g2b2(x, training=training)
        x = self.g3b1(x, training=training)
        x = self.g3b2(x, training=training)
        x = tf.nn.relu(self.bn_final(x, training=training))
        x = self.avgpool(x)
        x = self.flatten(x)
        return self.fc(x)


# --- Bottleneck block for ResNet-50 ---

class BottleneckBlock(tf.keras.layers.Layer):
    """Bottleneck residual block for ResNet-50."""
    def __init__(self, in_channels: int, mid_channels: int,
                 out_channels: int, stride: int = 1, **kwargs):
        super().__init__(**kwargs)
        self.conv1 = layers.Conv2D(mid_channels, 1, use_bias=False)
        self.bn1   = layers.BatchNormalization(momentum=BN_MOMENTUM, epsilon=BN_EPS)

        self.conv2 = layers.Conv2D(mid_channels, 3, strides=stride, padding="same", use_bias=False)
        self.bn2   = layers.BatchNormalization(momentum=BN_MOMENTUM, epsilon=BN_EPS)

        self.conv3 = layers.Conv2D(out_channels, 1, use_bias=False)
        self.bn3   = layers.BatchNormalization(momentum=BN_MOMENTUM, epsilon=BN_EPS)

        self.sc_conv = None
        self.sc_bn   = None
        if stride != 1 or in_channels != out_channels:
            self.sc_conv = layers.Conv2D(out_channels, 1, strides=stride, use_bias=False)
            self.sc_bn   = layers.BatchNormalization(momentum=BN_MOMENTUM, epsilon=BN_EPS)

    def call(self, x, training=False):
        if self.sc_conv is not None:
            sc = self.sc_bn(self.sc_conv(x), training=training)
        else:
            sc = x
        out = tf.nn.relu(self.bn1(self.conv1(x),   training=training))
        out = tf.nn.relu(self.bn2(self.conv2(out), training=training))
        out = tf.nn.relu(self.bn3(self.conv3(out), training=training))
        return tf.nn.relu(out + sc)


# --- ResNet-50 for Tiny ImageNet ---

class ResNet50TinyImageNet(tf.keras.Model):
    """ResNet-50 for Tiny ImageNet (64x64 inputs, 200 classes)."""
    def __init__(self, num_classes: int = 200):
        super().__init__()
        self.conv1   = layers.Conv2D(64, 3, padding="same", use_bias=True)
        self.bn1     = layers.BatchNormalization(momentum=BN_MOMENTUM, epsilon=BN_EPS)
        self.maxpool = layers.MaxPool2D(pool_size=3, strides=2, padding="same")

        self.layer1 = [
            BottleneckBlock(64,  64,  256, stride=1),
            BottleneckBlock(256, 64,  256, stride=1),
            BottleneckBlock(256, 64,  256, stride=1),
        ]
        self.layer2 = [
            BottleneckBlock(256, 128, 512, stride=2),
            BottleneckBlock(512, 128, 512, stride=1),
            BottleneckBlock(512, 128, 512, stride=1),
            BottleneckBlock(512, 128, 512, stride=1),
        ]
        self.layer3 = [
            BottleneckBlock(512,  256, 1024, stride=2),
            BottleneckBlock(1024, 256, 1024, stride=1),
            BottleneckBlock(1024, 256, 1024, stride=1),
            BottleneckBlock(1024, 256, 1024, stride=1),
            BottleneckBlock(1024, 256, 1024, stride=1),
            BottleneckBlock(1024, 256, 1024, stride=1),
        ]
        self.layer4 = [
            BottleneckBlock(1024, 512, 2048, stride=2),
            BottleneckBlock(2048, 512, 2048, stride=1),
            BottleneckBlock(2048, 512, 2048, stride=1),
        ]

        self.avgpool = layers.AveragePooling2D(pool_size=4, strides=1)
        self.flatten = layers.Flatten()
        self.fc      = layers.Dense(num_classes, use_bias=True)

    def call(self, x, training=False):
        x = tf.nn.relu(self.bn1(self.conv1(x), training=training))
        x = self.maxpool(x)
        for blk in self.layer1:
            x = blk(x, training=training)
        for blk in self.layer2:
            x = blk(x, training=training)
        for blk in self.layer3:
            x = blk(x, training=training)
        for blk in self.layer4:
            x = blk(x, training=training)
        x = self.avgpool(x)
        x = self.flatten(x)
        return self.fc(x)


# --- ResNet-50 for ImageNet-100 ---

class ResNet50ImageNet100(tf.keras.Model):
    """ResNet-50 for ImageNet-100 (224x224 inputs, 100 classes)."""
    def __init__(self, num_classes: int = 100):
        super().__init__()
        self.conv1   = layers.Conv2D(64, 7, strides=2, padding="same", use_bias=True)
        self.bn1     = layers.BatchNormalization(momentum=BN_MOMENTUM, epsilon=BN_EPS)
        self.maxpool = layers.MaxPool2D(pool_size=3, strides=2, padding="same")

        self.layer1 = [
            BottleneckBlock(64,  64,  256, stride=1),
            BottleneckBlock(256, 64,  256, stride=1),
            BottleneckBlock(256, 64,  256, stride=1),
        ]
        self.layer2 = [
            BottleneckBlock(256, 128, 512, stride=2),
            BottleneckBlock(512, 128, 512, stride=1),
            BottleneckBlock(512, 128, 512, stride=1),
            BottleneckBlock(512, 128, 512, stride=1),
        ]
        self.layer3 = [
            BottleneckBlock(512,  256, 1024, stride=2),
            BottleneckBlock(1024, 256, 1024, stride=1),
            BottleneckBlock(1024, 256, 1024, stride=1),
            BottleneckBlock(1024, 256, 1024, stride=1),
            BottleneckBlock(1024, 256, 1024, stride=1),
            BottleneckBlock(1024, 256, 1024, stride=1),
        ]
        self.layer4 = [
            BottleneckBlock(1024, 512, 2048, stride=2),
            BottleneckBlock(2048, 512, 2048, stride=1),
            BottleneckBlock(2048, 512, 2048, stride=1),
        ]

        self.avgpool = layers.AveragePooling2D(pool_size=7, strides=1)
        self.flatten = layers.Flatten()
        self.fc      = layers.Dense(num_classes, use_bias=True)

    def call(self, x, training=False):
        x = tf.nn.relu(self.bn1(self.conv1(x), training=training))
        x = self.maxpool(x)
        for blk in self.layer1:
            x = blk(x, training=training)
        for blk in self.layer2:
            x = blk(x, training=training)
        for blk in self.layer3:
            x = blk(x, training=training)
        for blk in self.layer4:
            x = blk(x, training=training)
        x = self.avgpool(x)
        x = self.flatten(x)
        return self.fc(x)


# --- GPT-2 small ---

class CausalSelfAttention(tf.keras.layers.Layer):
    """Multi-head causal self-attention."""
    def __init__(self, embed_dim: int, num_heads: int, dropout: float = 0.1, **kwargs):
        super().__init__(**kwargs)
        assert embed_dim % num_heads == 0
        self.num_heads = num_heads
        self.head_dim  = embed_dim // num_heads
        self.embed_dim = embed_dim

        init = tf.keras.initializers.TruncatedNormal(stddev=0.02)
        self.qkv        = layers.Dense(3 * embed_dim, use_bias=True,
                                       kernel_initializer=init,
                                       bias_initializer="zeros")
        self.proj       = layers.Dense(embed_dim, use_bias=True,
                                       kernel_initializer=init,
                                       bias_initializer="zeros")
        self.attn_drop  = layers.Dropout(dropout)
        self.resid_drop = layers.Dropout(dropout)

    def call(self, x, training=False):
        B = tf.shape(x)[0]
        T = tf.shape(x)[1]

        qkv = self.qkv(x)                                          # (B, T, 3C)
        q, k, v = tf.split(qkv, 3, axis=-1)                       # each (B, T, C)

        def split_heads(t):
            t = tf.reshape(t, [B, T, self.num_heads, self.head_dim])
            return tf.transpose(t, [0, 2, 1, 3])                  # (B, H, T, D)

        q, k, v = split_heads(q), split_heads(k), split_heads(v)

        # Scaled dot-product attention
        scale = tf.math.sqrt(tf.cast(self.head_dim, tf.float32))
        attn  = tf.matmul(q, k, transpose_b=True) / scale         # (B, H, T, T)

        # Additive causal mask: 0 for attended, -1e9 for future positions
        causal = tf.linalg.band_part(tf.ones([T, T], dtype=tf.bool), -1, 0)
        mask   = tf.cast(~causal, tf.float32) * -1e9               # (T, T)
        attn   = attn + mask                                       # broadcasts

        attn = tf.nn.softmax(attn, axis=-1)
        attn = self.attn_drop(attn, training=training)

        out = tf.matmul(attn, v)                                   # (B, H, T, D)
        out = tf.transpose(out, [0, 2, 1, 3])                     # (B, T, H, D)
        out = tf.reshape(out, [B, T, self.embed_dim])

        out = self.resid_drop(self.proj(out), training=training)
        return out


class GPTBlock(tf.keras.layers.Layer):
    """One GPT-2 transformer block."""
    def __init__(self, embed_dim: int, num_heads: int, ffn_dim: int,
                 dropout: float = 0.1, **kwargs):
        super().__init__(**kwargs)
        init = tf.keras.initializers.TruncatedNormal(stddev=0.02)
        self.ln_1     = layers.LayerNormalization(epsilon=1e-5)
        self.attn     = CausalSelfAttention(embed_dim, num_heads, dropout)
        self.ln_2     = layers.LayerNormalization(epsilon=1e-5)
        self.mlp_fc1  = layers.Dense(ffn_dim, use_bias=True,
                                     kernel_initializer=init,
                                     bias_initializer="zeros")
        self.mlp_fc2  = layers.Dense(embed_dim, use_bias=True,
                                     kernel_initializer=init,
                                     bias_initializer="zeros")
        self.ffn_drop = layers.Dropout(dropout)

    def call(self, x, training=False):
        x = x + self.attn(self.ln_1(x), training=training)
        h = self.mlp_fc2(tf.nn.gelu(self.mlp_fc1(self.ln_2(x))))
        x = x + self.ffn_drop(h, training=training)
        return x


class GPT2Small(tf.keras.Model):
    """GPT-2 small for causal language modelling."""
    def __init__(self,
                 vocab_size: int = 50257,
                 seq_len: int = 1024,
                 embed_dim: int = 768,
                 num_heads: int = 12,
                 num_layers: int = 12,
                 ffn_dim: int = 3072,
                 dropout: float = 0.1):
        super().__init__()
        self.seq_len    = seq_len
        self.embed_dim  = embed_dim
        self.vocab_size = vocab_size

        emb_init = tf.keras.initializers.TruncatedNormal(stddev=0.02)
        self.token_embed = layers.Embedding(vocab_size, embed_dim,
                                            embeddings_initializer=emb_init)
        self.drop   = layers.Dropout(dropout)
        self.blocks = [GPTBlock(embed_dim, num_heads, ffn_dim, dropout)
                       for _ in range(num_layers)]
        self.ln_f   = layers.LayerNormalization(epsilon=1e-5)

        # pos_embed and head_bias initialized in build()
        self.pos_embed = None
        self.head_bias = None

    def build(self, input_shape):
        pos_init = tf.keras.initializers.TruncatedNormal(stddev=0.01)
        self.pos_embed = self.add_weight(
            name="pos_embed",
            shape=(1, self.seq_len, self.embed_dim),
            initializer=pos_init,
            trainable=True,
        )
        self.head_bias = self.add_weight(
            name="head_bias",
            shape=(self.vocab_size,),
            initializer="zeros",
            trainable=True,
        )
        super().build(input_shape)

    def call(self, x, training=False):
        T = tf.shape(x)[1]
        tok = self.token_embed(x)                      # (B, T, embed_dim)
        pos = self.pos_embed[:, :T, :]                 # (1, T, embed_dim)
        x   = self.drop(tok + pos, training=training)

        for block in self.blocks:
            x = block(x, training=training)

        x = self.ln_f(x)
        # Weight-tied output projection (shared with token_embed)
        logits = tf.matmul(x, tf.transpose(self.token_embed.embeddings)) + self.head_bias
        return logits


# ======================== Model Registry ========================

def get_model_config(model_name: str) -> Dict[str, Any]:
    """
    Returns configuration dictionary for the specified model.

    Each config contains:
        - model_cls:        Model class constructor
        - train_dataset:    Training dataset factory -> (tf.data.Dataset, num_samples)
        - test_dataset:     Test/validation dataset factory
        - epochs:           Number of training epochs
        - batch_size:       Batch size
        - lr:               Initial learning rate
        - num_workers:      Parallel map workers for image loading
        - optimizer_type:   'adam' or 'adamw'
        - scheduler_type:   'step' or 'cosine'
        - is_language_model: bool flag for GPT-2
        - input_shape:      (H, W, C) for image models, None for LM
    """
    configs = {
        "resnet9_cifar10": {
            "model_cls":    lambda: ResNet9CIFAR10(num_classes=10),
            "train_dataset": lambda bs, nw: make_cifar10_dataset(
                os.getenv("CIFAR10_BIN_ROOT", "data/cifar-10-batches-bin"),
                train=True, batch_size=bs, num_workers=nw,
            ),
            "test_dataset": lambda bs, nw: make_cifar10_dataset(
                os.getenv("CIFAR10_BIN_ROOT", "data/cifar-10-batches-bin"),
                train=False, batch_size=bs, num_workers=nw,
            ),
            "epochs":           int(os.getenv("EPOCHS", "10")),
            "batch_size":       int(os.getenv("BATCH_SIZE", "128")),
            "lr":               float(os.getenv("LR_INITIAL", "0.001")),
            "num_workers":      2,
            "optimizer_type":   "adam",
            "scheduler_type":   "step",
            "is_language_model": False,
            "input_shape":      (32, 32, 3),
        },

        "wrn16_8_cifar100": {
            "model_cls":    lambda: WRN16_8CIFAR100(num_classes=100),
            "train_dataset": lambda bs, nw: make_cifar100_dataset(
                os.getenv("CIFAR100_BIN_ROOT", "data/cifar-100-binary"),
                train=True, batch_size=bs, num_workers=nw,
            ),
            "test_dataset": lambda bs, nw: make_cifar100_dataset(
                os.getenv("CIFAR100_BIN_ROOT", "data/cifar-100-binary"),
                train=False, batch_size=bs, num_workers=nw,
            ),
            "epochs":           int(os.getenv("EPOCHS", "50")),
            "batch_size":       int(os.getenv("BATCH_SIZE", "128")),
            "lr":               float(os.getenv("LR_INITIAL", "0.001")),
            "num_workers":      2,
            "optimizer_type":   "adam",
            "scheduler_type":   "step",
            "is_language_model": False,
            "input_shape":      (32, 32, 3),
        },

        "resnet50_tiny_imagenet": {
            "model_cls":    lambda: ResNet50TinyImageNet(num_classes=200),
            "train_dataset": lambda bs, nw: make_tiny_imagenet_dataset(
                os.getenv("TINY_IMAGENET_ROOT", "data/tiny-imagenet-200"),
                train=True, batch_size=bs, num_workers=nw,
            ),
            "test_dataset": lambda bs, nw: make_tiny_imagenet_dataset(
                os.getenv("TINY_IMAGENET_ROOT", "data/tiny-imagenet-200"),
                train=False, batch_size=bs, num_workers=nw,
            ),
            "epochs":           int(os.getenv("EPOCHS", "90")),
            "batch_size":       int(os.getenv("BATCH_SIZE", "128")),
            "lr":               float(os.getenv("LR_INITIAL", "0.001")),
            "num_workers":      4,
            "optimizer_type":   "adam",
            "scheduler_type":   "step",
            "is_language_model": False,
            "input_shape":      (64, 64, 3),
        },

        "resnet50_imagenet100": {
            "model_cls":    lambda: ResNet50ImageNet100(num_classes=100),
            "train_dataset": lambda bs, nw: make_imagenet100_dataset(
                os.getenv("IMAGENET100_ROOT", "data/imagenet-100"),
                train=True, batch_size=bs, num_workers=nw,
            ),
            "test_dataset": lambda bs, nw: make_imagenet100_dataset(
                os.getenv("IMAGENET100_ROOT", "data/imagenet-100"),
                train=False, batch_size=bs, num_workers=nw,
            ),
            "epochs":           int(os.getenv("EPOCHS", "90")),
            "batch_size":       int(os.getenv("BATCH_SIZE", "64")),
            "lr":               float(os.getenv("LR_INITIAL", "0.001")),
            "num_workers":      4,
            "optimizer_type":   "adam",
            "scheduler_type":   "step",
            "is_language_model": False,
            "input_shape":      (224, 224, 3),
        },

        "gpt2_small": {
            "model_cls": lambda: GPT2Small(
                vocab_size=50257, seq_len=1024,
                embed_dim=768, num_heads=12,
                num_layers=12, ffn_dim=3072, dropout=0.1,
            ),
            "train_dataset": lambda bs, nw: make_openwebtext_dataset(
                path=os.getenv("OPENWEBTEXT_PATH", "data/open-web-text/train.bin"),
                seq_len=1024, train=True, batch_size=bs, num_workers=nw,
            ),
            "test_dataset": lambda bs, nw: make_openwebtext_dataset(
                path=os.getenv("OPENWEBTEXT_VAL_PATH", "data/open-web-text/val.bin"),
                seq_len=1024, train=False, batch_size=bs, num_workers=nw,
            ),
            "epochs":           int(os.getenv("EPOCHS", "1")),
            "batch_size":       int(os.getenv("BATCH_SIZE", "8")),
            "lr":               float(os.getenv("LR_INITIAL", "3e-4")),
            "num_workers":      2,
            "optimizer_type":   "adamw",
            "scheduler_type":   "cosine",
            "is_language_model": True,
            "input_shape":      None,
        },
    }

    if model_name not in configs:
        raise ValueError(f"Unknown model: {model_name}. Available: {list(configs.keys())}")

    return configs[model_name]


# ======================== Training ========================

def train_epoch(model, train_ds, optimizer, loss_fn, cfg, epoch,
                num_batches, batch_writer):
    """Train for one epoch using tf.GradientTape."""
    running_loss    = 0.0
    running_correct = 0.0
    running_total   = 0.0
    is_lm = cfg["is_language_model"]

    @tf.function
    def train_step(inputs, targets):
        with tf.GradientTape() as tape:
            outputs = model(inputs, training=True)
            if is_lm:
                B  = tf.shape(outputs)[0]
                T  = tf.shape(outputs)[1]
                V  = tf.shape(outputs)[2]
                flat_out = tf.reshape(outputs, [B * T, V])
                flat_tgt = tf.reshape(targets, [B * T])
                loss     = loss_fn(flat_tgt, flat_out)
                predicted = tf.argmax(flat_out, axis=-1)
                correct   = tf.cast(
                    tf.reduce_sum(tf.cast(
                        tf.equal(predicted, tf.cast(flat_tgt, tf.int64)), tf.float32
                    )), tf.float32
                )
                total = tf.cast(B * T, tf.float32)
            else:
                loss      = loss_fn(targets, outputs)
                predicted = tf.argmax(outputs, axis=-1)
                correct   = tf.cast(
                    tf.reduce_sum(tf.cast(
                        tf.equal(predicted, tf.cast(targets, tf.int64)), tf.float32
                    )), tf.float32
                )
                total = tf.cast(tf.shape(targets)[0], tf.float32)

        grads = tape.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(zip(grads, model.trainable_variables))
        return loss, correct, total

    for batch_idx, (inputs, targets) in enumerate(train_ds):
        step_start = time.time()
        loss, correct, total = train_step(inputs, targets)
        step_ms = int((time.time() - step_start) * 1000)

        loss_val    = float(loss.numpy())
        correct_val = float(correct.numpy())
        total_val   = float(total.numpy())

        running_loss    += loss_val * total_val
        running_correct += correct_val
        running_total   += total_val

        batch_acc = 100.0 * correct_val / total_val
        batch_writer.writerow([
            epoch, batch_idx + 1,
            f"{loss_val:.6f}",
            f"{batch_acc:.4f}",
            step_ms,
        ])

        if (batch_idx + 1) % 100 == 0:
            print(
                f"[Train Batch {batch_idx + 1}/{num_batches}] "
                f"Loss: {loss_val:.4f} | Acc: {batch_acc:.2f}% | "
                f"Step time: {step_ms}ms"
            )

    train_loss = running_loss    / running_total
    train_acc  = 100.0 * running_correct / running_total
    return train_loss, train_acc


def validate(model, val_ds, loss_fn, cfg, epoch, val_writer):
    """Validate the model."""
    val_loss_sum = 0.0
    val_correct  = 0.0
    val_total    = 0.0
    is_lm = cfg["is_language_model"]

    @tf.function
    def val_step(inputs, targets):
        outputs = model(inputs, training=False)
        if is_lm:
            B  = tf.shape(outputs)[0]
            T  = tf.shape(outputs)[1]
            V  = tf.shape(outputs)[2]
            flat_out  = tf.reshape(outputs, [B * T, V])
            flat_tgt  = tf.reshape(targets, [B * T])
            loss      = loss_fn(flat_tgt, flat_out)
            predicted = tf.argmax(flat_out, axis=-1)
            correct   = tf.cast(
                tf.reduce_sum(tf.cast(
                    tf.equal(predicted, tf.cast(flat_tgt, tf.int64)), tf.float32
                )), tf.float32
            )
            total = tf.cast(B * T, tf.float32)
        else:
            loss      = loss_fn(targets, outputs)
            predicted = tf.argmax(outputs, axis=-1)
            correct   = tf.cast(
                tf.reduce_sum(tf.cast(
                    tf.equal(predicted, tf.cast(targets, tf.int64)), tf.float32
                )), tf.float32
            )
            total = tf.cast(tf.shape(targets)[0], tf.float32)
        return loss, correct, total

    for val_step_idx, (inputs, targets) in enumerate(val_ds):
        loss, correct, total = val_step(inputs, targets)

        loss_val    = float(loss.numpy())
        correct_val = float(correct.numpy())
        total_val   = float(total.numpy())

        val_loss_sum += loss_val * total_val
        val_correct  += correct_val
        val_total    += total_val

        step_acc = 100.0 * correct_val / total_val
        val_writer.writerow([
            epoch, val_step_idx + 1,
            f"{loss_val:.6f}",
            f"{step_acc:.4f}",
        ])

    val_loss = val_loss_sum / val_total
    val_acc  = 100.0 * val_correct / val_total
    return val_loss, val_acc


def main():
    parser = argparse.ArgumentParser(description="Unified SYNET TensorFlow Trainer")
    parser.add_argument("--model", type=str, required=True,
                        choices=["resnet9_cifar10", "wrn16_8_cifar100",
                                 "resnet50_tiny_imagenet", "resnet50_imagenet100",
                                 "gpt2_small"],
                        help="Model to train")
    parser.add_argument("--epochs",     type=int,   default=None,
                        help="Number of epochs (overrides config default)")
    parser.add_argument("--batch-size", type=int,   default=None,
                        help="Batch size (overrides config default)")
    parser.add_argument("--lr",         type=float, default=None,
                        help="Learning rate (overrides config default)")
    parser.add_argument("--gpu",        type=int,   default=0,
                        help="GPU index to use (default: 0); -1 for CPU")
    args = parser.parse_args()

    # Get model configuration
    cfg = get_model_config(args.model)

    # Override config with command-line arguments
    if args.epochs is not None:
        cfg["epochs"] = args.epochs
    if args.batch_size is not None:
        cfg["batch_size"] = args.batch_size
    if args.lr is not None:
        cfg["lr"] = args.lr

    # Device setup
    gpus = tf.config.list_physical_devices("GPU")
    if args.gpu >= 0 and gpus:
        tf.config.set_visible_devices(gpus[args.gpu], "GPU")
        tf.config.experimental.set_memory_growth(gpus[args.gpu], True)
        device_str = f"GPU:{args.gpu}"
    else:
        tf.config.set_visible_devices([], "GPU")
        device_str = "CPU"

    print(f">>> Running on device: {device_str}")
    print(f">>> Model: {args.model}")
    print(f">>> Epochs: {cfg['epochs']}")
    print(f">>> Batch size: {cfg['batch_size']}")
    print(f">>> Learning rate: {cfg['lr']}")

    # Load datasets
    print(">>> Loading datasets...")
    train_ds, num_train = cfg["train_dataset"](cfg["batch_size"], cfg["num_workers"])
    val_ds,   num_val   = cfg["test_dataset"](cfg["batch_size"],  cfg["num_workers"])
    print(f">>> Train samples: {num_train}")
    print(f">>> Val samples:   {num_val}")

    num_batches = math.ceil(num_train / cfg["batch_size"])

    # Create and build model
    print(">>> Creating model...")
    model = cfg["model_cls"]()

    if cfg["input_shape"] is not None:
        dummy = tf.zeros([1, *cfg["input_shape"]])
    else:
        dummy = tf.zeros([1, 1024], dtype=tf.int64)
    _ = model(dummy, training=False)

    total_params = int(sum(tf.size(v).numpy() for v in model.trainable_variables))
    print(f">>> Parameters: {total_params:,}")

    # Loss function (equivalent to nn.CrossEntropyLoss)
    loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(
        from_logits=True, reduction="sum_over_batch_size"
    )

    # Optimizer
    if cfg["optimizer_type"] == "adamw":
        try:
            optimizer = tf.keras.optimizers.AdamW(
                learning_rate=cfg["lr"],
                weight_decay=0.1,
                beta_1=0.9,
                beta_2=0.95,
                epsilon=1e-8,
            )
        except AttributeError:
            # Older TF fallback
            optimizer = tf.keras.optimizers.experimental.AdamW(
                learning_rate=cfg["lr"],
                weight_decay=0.1,
                beta_1=0.9,
                beta_2=0.95,
                epsilon=1e-8,
            )
    else:
        optimizer = tf.keras.optimizers.Adam(
            learning_rate=cfg["lr"],
            beta_1=0.9,
            beta_2=0.999,
            epsilon=1e-3,
        )

    # Logging setup
    ts      = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    log_dir = "logs"
    os.makedirs(log_dir, exist_ok=True)

    batch_csv_path = os.path.join(log_dir, f"{args.model}_batch_{ts}.csv")
    epoch_csv_path = os.path.join(log_dir, f"{args.model}_epoch_{ts}.csv")
    val_csv_path   = os.path.join(log_dir, f"{args.model}_val_{ts}.csv")

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

    # Training loop
    print(f"\n>>> Starting training for {cfg['epochs']} epochs...")
    for epoch in range(1, cfg["epochs"] + 1):
        print(f"\n===== Epoch {epoch}/{cfg['epochs']} =====")
        epoch_start = time.time()

        # Update learning rate (StepLR: decay by 0.1 every 5 epochs)
        if cfg["scheduler_type"] == "step":
            new_lr = cfg["lr"] * (0.1 ** ((epoch - 1) // 5))
            optimizer.learning_rate = new_lr
        # cosine: LR unchanged (mirrors PyTorch reference behaviour)

        # Train
        train_loss, train_acc = train_epoch(
            model, train_ds, optimizer, loss_fn, cfg,
            epoch, num_batches, batch_writer,
        )
        batch_csv_file.flush()

        # Validate
        val_loss, val_acc = validate(
            model, val_ds, loss_fn, cfg, epoch, val_writer,
        )
        val_csv_file.flush()

        epoch_time = time.time() - epoch_start

        # Log epoch summary
        epoch_writer.writerow([
            epoch,
            f"{train_loss:.6f}", f"{train_acc:.4f}",
            f"{val_loss:.6f}",   f"{val_acc:.4f}",
        ])
        epoch_csv_file.flush()

        print(
            f"Epoch {epoch}/{cfg['epochs']} completed in {epoch_time:.2f}s\n"
            f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}%\n"
            f"Val   Loss: {val_loss:.4f} | Val   Acc: {val_acc:.2f}%"
        )

        if cfg["is_language_model"]:
            perplexity = math.exp(min(val_loss, 20))
            print(f"Val Perplexity: {perplexity:.2f}")

    # Cleanup
    batch_csv_file.close()
    epoch_csv_file.close()
    val_csv_file.close()

    print(f"\n>>> Logs saved to {log_dir}/{args.model}_*_{ts}.csv")
    print(f">>> Training completed for {args.model}!")


if __name__ == "__main__":
    main()
