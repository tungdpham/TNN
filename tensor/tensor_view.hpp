enum Layout {
  NCHW,  // 4D: Batch, Channels, Height, Width (most common for CNNs)
  NHWC,  // 4D: Batch, Height, Width, Channels (TensorFlow default)
  NCDHW, // 5D: Batch, Channels, Depth, Height, Width (3D CNNs)
  NDHWC  // 5D: Batch, Depth, Height, Width, Channels (3D TensorFlow default)
};

template <typename T, Layout L> struct TensorView;

template <typename T> struct TensorView<T, NCHW> {
  static constexpr size_t dims = 4;

  inline static void compute_strides(size_t *strides, const size_t *shape) {
    strides[0] = shape[1] * shape[2] * shape[3]; // stride for batch
    strides[1] = shape[2] * shape[3];            // stride for channels
    strides[2] = shape[3];                       // stride for height
    strides[3] = 1;                              // stride for width
  }
};

template <typename T> struct TensorView<T, NHWC> {
  static constexpr size_t dims = 4;

  inline static void compute_strides(size_t *strides, const size_t *shape) {
    strides[0] = shape[2] * shape[3] * shape[1]; // stride for batch
    strides[1] = shape[3] * shape[1];            // stride for height
    strides[2] = shape[1];                       // stride for width
    strides[3] = 1;                              // stride for channels
  }
};

template <typename T> struct TensorView<T, NCDHW> {
  static constexpr size_t dims = 5;

  inline static void compute_strides(size_t *strides, const size_t *shape) {
    strides[0] = shape[1] * shape[2] * shape[3] * shape[4]; // stride for batch
    strides[1] = shape[2] * shape[3] * shape[4]; // stride for channels
    strides[2] = shape[3] * shape[4];            // stride for depth
    strides[3] = shape[4];                       // stride for height
    strides[4] = 1;                              // stride for width
  }
};

template <typename T> struct TensorView<T, NDHWC> {
  static constexpr size_t dims = 5;

  inline static void compute_strides(size_t *strides, const size_t *shape) {
    strides[0] = shape[2] * shape[3] * shape[4] * shape[1]; // stride for batch
    strides[1] = shape[3] * shape[4] * shape[1];            // stride for depth
    strides[2] = shape[4] * shape[1];                       // stride for height
    strides[3] = shape[1];                                  // stride for width
    strides[4] = 1; // stride for channels
  }
};