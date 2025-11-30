# Tensor Usage Guide

This guide covers the implementation of `Tensor` class and how to use it the to store and access data.

## Table of Contents
- [Storage Type](#storage-type)
- [Creating a Tensor](#creating-a-tensor)

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
- `Tensor(std::vector<size_t> shape, const device_ptr<T[]> &data)`: Create a tensor with external data pointer.

## Accessing Elements
To 
Elements are accessed using the `operator()` with dimension indices:
```c
T &value = tensor(n, c, h, w); // For NCHW layout
```
Note: Since the CPU cannot dereference memory on GPU's pinned memory. You have to ensure the tensor is on host memory.