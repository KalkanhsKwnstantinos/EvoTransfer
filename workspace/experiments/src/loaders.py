from keras.datasets import cifar10, cifar100, mnist, fashion_mnist
import tensorflow as tf
import numpy as np

#tf.keras.mixed_precision.set_global_policy('mixed_float16')

def load_mnist():
    
    (x_train, y_train), (x_val, y_val) = mnist.load_data()

    x_train = (x_train.astype("float32") / 127.5 - 1.0)[..., None]
    x_val = (x_val.astype("float32") / 127.5 - 1.0)[..., None]

    train_ds = tf.data.Dataset.from_tensor_slices((x_train, y_train))
    val_ds = tf.data.Dataset.from_tensor_slices((x_val, y_val))
    
    return (train_ds, val_ds), (x_val.shape[1:], len(np.unique(y_val)))
    
def load_fashion_mnist():
    
    (x_train, y_train), (x_val, y_val) = fashion_mnist.load_data()

    x_train = (x_train.astype("float32") / 127.5 - 1.0)[..., None]
    x_val = (x_val.astype("float32") / 127.5 - 1.0)[..., None]
    
    train_ds = tf.data.Dataset.from_tensor_slices((x_train, y_train))
    val_ds = tf.data.Dataset.from_tensor_slices((x_val, y_val))
    
    return (train_ds, val_ds), (x_val.shape[1:], len(np.unique(y_val)))    
    
def load_cifar10():
    
    (x_train, y_train), (x_val, y_val) = cifar10.load_data()
    
    x_train = x_train.astype("float32") / 255.0
    x_val = x_val.astype("float32") / 255.0

    mean = x_train.mean(axis=(0,1,2), keepdims=True)
    std  = x_train.std(axis=(0,1,2), keepdims=True) + 1e-7

    x_train = (x_train - mean) / std
    x_test  = (x_test - mean) / std

    train_ds = tf.data.Dataset.from_tensor_slices((x_train, y_train))
    val_ds = tf.data.Dataset.from_tensor_slices((x_val, y_val))
    
    return (train_ds, val_ds), (x_val.shape[1:], len(np.unique(y_val)))
    
def load_cifar100():
    
    (x_train, y_train), (x_val, y_val) = cifar100.load_data()
    
    x_train = x_train.astype("float32") / 255.0
    x_val = x_val.astype("float32") / 255.0

    mean = x_train.mean(axis=(0,1,2), keepdims=True)
    std  = x_train.std(axis=(0,1,2), keepdims=True) + 1e-7

    x_train = (x_train - mean) / std
    x_test  = (x_test - mean) / std

    train_ds = tf.data.Dataset.from_tensor_slices((x_train, y_train))
    val_ds = tf.data.Dataset.from_tensor_slices((x_val, y_val))
    
    return (train_ds, val_ds), (x_val.shape[1:], len(np.unique(y_val)))