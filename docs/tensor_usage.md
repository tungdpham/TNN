# Tensor Usage Guide

This guide covers the implementation of `Tensor` class and how to use it the to store and access data.

## Table of Contents
- [Storage Type](#storage-type)
- [Creating a Tensor](#creating-a-tensor)
- [Accessing Elements](#accessing-elements)
- [Arithmetic Operations](#arithmetic-operations)
- [Shape and Dimensions](#shape-and-dimensions)
- [Data Manipulation](#data-manipulation)
- [Reshaping and Slicing](#reshaping-and-slicing)
- [Statistical Operations](#statistical-operations)
- [Serialization](#serialization)
- [Other Methods](#other-methods)

## Storage Type
The tensor structure uses a 1d-array to store the data and a layout trait struct that tells the systems how to intepret it. For example, NCHW layout will have a corresponding layout metadata that tells the system to treat the 1d array as a row-major with dimensions of (batch size, channels, height, width). Most methods primarily supports row-major. Column major and channel major are to be implemented. 

## Creating a Tensor
There are several ways to construct a Tensor. The most common one is to use

```c
Tensor<float, NCHW> tensor(64, 32, 48, 48); // Create a tensor with NCHW layout that stores single precision floats. The dimensions are 64, 32, 48, 48 respectively for batch size, channels, height, and width.
```

Other constructors include:
- `Tensor()`: Default constructor creating an empty tensor.
- `Tensor(std::vector<size_t> shape)`: Create a tensor with specified shape vector.
- `Tensor(size_t batch_size, size_t channels, size_t height, size_t width, T *data)`: Create a 4D tensor with external data pointer.
- `Tensor(std::vector<size_t> shape, const T *data)`: Create a tensor with shape and external data.

## Accessing Elements
Elements are accessed using the `operator()` with dimension indices:
```c
T &value = tensor(n, c, h, w); // For NCHW layout
```

## Arithmetic Operations
Tensors support element-wise arithmetic operations:
- `tensor1 + tensor2`: Element-wise addition.
- `tensor1 - tensor2`: Element-wise subtraction.
- `tensor1 * tensor2`: Element-wise multiplication.
- `tensor1 / tensor2`: Element-wise division.
- `tensor * scalar`: Scalar multiplication.
- `tensor / scalar`: Scalar division.
- In-place versions: `+=`, `-=`, `*=`, `/=`.

## Shape and Dimensions
- `shape()`: Returns a vector of dimensions.
- `shape_str()`: Returns a string representation of the shape.
- `batch_size()`, `channels()`, `height()`, `width()`, `depth()`: Access specific dimensions.
- `dimension(size_t index)`: Get dimension at index.
- `stride(size_t index)`: Get stride at index.
- `size()`: Total number of elements.
- `same_shape(const Tensor &other)`: Check if shapes match.

## Data Manipulation
- `data()`: Get raw data pointer.
- `is_aligned(size_t alignment)`: Check memory alignment.
- `clone()`: Create a deep copy.
- `fill(T value)`: Fill tensor with a constant value.
- `fill_random_uniform(T range)`: Fill with uniform random values.
- `fill_random_normal(T mean, T stddev)`: Fill with normal distribution values.

## Reshaping and Slicing
- `reshape(const std::vector<size_t> &new_shape)`: Reshape to new dimensions (total size must match).
- `pad(size_t pad_h, size_t pad_w, T value)`: Pad height and width with value.
- `unpad(size_t pad_h, size_t pad_w)`: Remove padding.
- `crop(size_t start_h, size_t start_w, size_t end_h, size_t end_w)`: Crop to specified region.
- `slice_batch(size_t start_batch, size_t end_batch)`: Slice along batch dimension.
- `slice_channels(size_t start_ch, size_t end_ch)`: Slice along channel dimension.
- `copy_batch(Tensor &other, size_t src_batch_idx, size_t dest_batch_idx)`: Copy a batch from another tensor.

## Statistical Operations
- `mean()`: Compute overall mean.
- `variance()`: Compute overall variance.
- `channel_means()`: Compute mean per channel.
- `apply_softmax()`: Apply softmax activation in-place.

## Serialization
- `save(std::ofstream &out)`: Save tensor to binary stream.
- `load(std::ifstream &in)`: Load tensor from binary stream.

## Other Methods
- `split(size_t num_splits)`: Split tensor into multiple tensors along batch dimension.
- `transpose<New_Layout>()`: Transpose to a new layout (template method).
- `print_data()`: Print tensor data to console.
