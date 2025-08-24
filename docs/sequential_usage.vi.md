# Hướng Dẫn Sử Dụng Sequential Model

Hướng dẫn này bao gồm cách sử dụng lớp `Sequential` và `SequentialBuilder` để xây dựng và huấn luyện các mô hình mạng neural.

## Mục Lục
- [Sử Dụng Cơ Bản Lớp Sequential](#sử-dụng-cơ-bản-lớp-sequential)
- [SequentialBuilder Với Tự Động Suy Luận Hình Dạng](#sequentialbuilder-với-tự-động-suy-luận-hình-dạng)
- [Cấu Hình Huấn Luyện](#cấu-hình-huấn-luyện)
- [Huấn Luyện Mô Hình](#huấn-luyện-mô-hình)
- [Lưu Trữ Mô Hình](#lưu-trữ-mô-hình)
- [Theo Dõi Hiệu Suất](#theo-dõi-hiệu-suất-tùy-chọn)

## Sử Dụng Cơ Bản Lớp Sequential

### Tạo Mô Hình
```cpp
// Tạo mô hình sequential rỗng
tnn::Sequential<float> model("ten_mo_hinh_cua_toi");

// Thêm các lớp thủ công
model.add(std::make_unique<DenseLayer<float>>(784, 128, "relu"));
model.add(std::make_unique<DenseLayer<float>>(128, 10, "linear"));
```

### Quản Lý Lớp
```cpp
// Chèn lớp tại vị trí cụ thể
model.insert(1, std::make_unique<DropoutLayer<float>>(0.5));

// Xóa lớp theo chỉ số
model.remove(2);

// Truy cập các lớp
Layer<float>& layer = model[0];  // hoặc model.at(0)

// Lấy kích thước mô hình
size_t so_lop = model.size();
```

## SequentialBuilder Với Tự Động Suy Luận Hình Dạng

`SequentialBuilder` cung cấp giao diện linh hoạt với tự động suy luận hình dạng, loại bỏ việc phải tính toán thủ công kích thước đầu vào cho mỗi lớp.

### Mẫu Sử Dụng Cơ Bản
```cpp
auto model = tnn::SequentialBuilder<float>("ten_mo_hinh")
    .input({kenh, chieu_cao, chieu_rong})  // Đặt hình dạng đầu vào trước
    .conv2d(kenh_dau_ra, kernel_h, kernel_w, ...)  // Tự động suy luận kênh đầu vào
    .maxpool2d(pool_h, pool_w, ...)
    .flatten()
    .dense(dac_trung_dau_ra, ...)  // Tự động suy luận đặc trưng đầu vào
    .build();
```

### Các Phương Thức Lớp

#### Lớp Đầu Vào (Bắt Buộc Đầu Tiên)
```cpp
.input({1, 28, 28})  // Hình dạng không bao gồm chiều batch: [kênh, chiều cao, chiều rộng]
```
**Tham số:**
- `shape`: Kích thước đầu vào không bao gồm kích thước batch

#### Lớp Tích Chập
```cpp
// Tự động suy luận kênh đầu vào
.conv2d(kenh_dau_ra, kernel_h, kernel_w, stride_h, stride_w, pad_h, pad_w, kich_hoat, su_dung_bias, ten)

// Chỉ định thủ công (tương thích ngược)
.conv2d(kenh_dau_vao, kenh_dau_ra, kernel_h, kernel_w, ...)
```
**Tham số:**
- `kenh_dau_ra`: Số lượng bản đồ đặc trưng đầu ra
- `kernel_h/w`: Kích thước kernel
- `stride_h/w`: Giá trị stride (mặc định: 1)
- `pad_h/w`: Giá trị padding (mặc định: 0)
- `kich_hoat`: Hàm kích hoạt ("relu", "elu", "linear", v.v.)
- `su_dung_bias`: Có sử dụng bias không (mặc định: true)
- `ten`: Tên lớp (tùy chọn, tự động tạo nếu trống)

#### Lớp Pooling
```cpp
.maxpool2d(pool_h, pool_w, stride_h, stride_w, pad_h, pad_w, ten)
```
**Tham số:**
- `pool_h/w`: Kích thước cửa sổ pooling
- `stride_h/w`: Giá trị stride (mặc định: giống kích thước pool)
- `pad_h/w`: Giá trị padding (mặc định: 0)

#### Lớp Dense
```cpp
// Tự động suy luận đặc trưng đầu vào
.dense(dac_trung_dau_ra, kich_hoat, su_dung_bias, ten)

// Chỉ định thủ công
.dense(dac_trung_dau_vao, dac_trung_dau_ra, kich_hoat, su_dung_bias, ten)
```
**Tham số:**
- `dac_trung_dau_ra`: Số lượng neuron đầu ra
- `kich_hoat`: Hàm kích hoạt
- `su_dung_bias`: Có sử dụng bias không (mặc định: true)

#### Lớp Tiện Ích
```cpp
.flatten(ten)                    // Làm phẳng đầu vào nhiều chiều
.dropout(ty_le_dropout, ten)      // Regularization dropout
.activation(ten_kich_hoat, ten) // Lớp kích hoạt độc lập
.batchnorm(epsilon, momentum, affine, ten)  // Batch normalization
```

## Cấu Hình Huấn Luyện

### Thiết Lập Optimizer
```cpp
// Tạo và thiết lập optimizer
auto optimizer = std::make_unique<tnn::Adam<float>>(
    ty_le_hoc,    // ví dụ: 0.001f
    beta1,        // ví dụ: 0.9f (hệ số momentum)
    beta2,        // ví dụ: 0.999f (hệ số RMSprop)
    epsilon       // ví dụ: 1e-8f (ổn định số)
);
model.set_optimizer(std::move(optimizer));
```

### Thiết Lập Hàm Loss
```cpp
// Tạo và thiết lập hàm loss
auto loss = tnn::LossFactory<float>::create_crossentropy(epsilon); // ví dụ: 1e-15f
model.set_loss_function(std::move(loss));
```

### Điều Khiển Chế Độ Huấn Luyện
```cpp
model.train();  // Bật chế độ huấn luyện (ảnh hưởng dropout, batchnorm)
model.eval();   // Bật chế độ đánh giá
```

## Huấn Luyện Mô Hình

### Ví Dụ Vòng Lặp Huấn Luyện
```cpp
for (int epoch = 0; epoch < so_epoch; ++epoch) {
    model.train();
    
    while (data_loader.get_next_batch(du_lieu_batch, nhan_batch)) {
        // Lan truyền tiến
        Tensor<float> du_doan = model.forward(du_lieu_batch);
        
        // Áp dụng softmax cho phân loại
        utils::apply_softmax<float>(du_doan);
        
        // Tính loss (để theo dõi)
        float loss = model.loss_function()->compute_loss(du_doan, nhan_batch);
        
        // Lan truyền ngược
        Tensor<float> gradient_loss = 
            model.loss_function()->compute_gradient(du_doan, nhan_batch);
        model.backward(gradient_loss);
        
        // Cập nhật tham số
        model.update_parameters();
    }
    
    // Giai đoạn validation
    model.eval();
    // ... mã validation ...
}
```

## Lưu Trữ Mô Hình

### Lưu Mô Hình
```cpp
// Lưu cả kiến trúc và trọng số
model.save_to_file("duong_dan/den/mo_hinh");  // Tạo các file .json và .bin

// Chỉ lưu cấu hình
model.save_config("config.json");
```

### Tải Mô Hình
```cpp
// Tải mô hình hoàn chỉnh
auto mo_hinh_da_tai = tnn::Sequential<float>::from_file("duong_dan/den/mo_hinh");

// Tải từ cấu hình
auto mo_hinh_config = tnn::Sequential<float>::load_from_config_file("config.json");
```

## Theo Dõi Hiệu Suất (Tùy Chọn)

### Profiling
```cpp
// Bật profiling hiệu suất
model.enable_profiling(true);

// Sau huấn luyện, xem kết quả thời gian tích lũy
model.print_profiling_summary();

// Xóa dữ liệu profiling
model.clear_profiling_data();
```

### Thông Tin Mô Hình
```cpp
// In tóm tắt kiến trúc mô hình
model.print_summary({kich_thuoc_batch, kenh, chieu_cao, chieu_rong});

// Lấy số lượng tham số
size_t tong_tham_so = model.parameter_count();

// In cấu hình mô hình
model.print_config();
```

### Lập Lịch Tỷ Lệ Học
```cpp
// Lấy tỷ lệ học hiện tại
float ty_le_hoc_hien_tai = model.optimizer()->get_learning_rate();

// Cập nhật tỷ lệ học
model.optimizer()->set_learning_rate(ty_le_hoc_moi);
```

## Ví Dụ Hoàn Chỉnh

```cpp
#include "nn/sequential.hpp"
#include "nn/optimizers.hpp"
#include "nn/loss.hpp"

// Xây dựng mô hình với tự động suy luận hình dạng
auto model = tnn::SequentialBuilder<float>("phan_loai_mnist")
    .input({1, 28, 28})                           // 1 kênh, ảnh 28x28
    .conv2d(32, 5, 5, 1, 1, 0, 0, "relu")        // 32 bộ lọc, kernel 5x5, stride 1x1, padding 0x0
    .maxpool2d(2, 2, 2, 2, 0, 0)                 // pool 2x2, stride 2x2, padding 0x0
    .conv2d(64, 5, 5, 1, 1, 0, 0, "relu")        // 64 bộ lọc, kernel 5x5, stride 1x1, padding 0x0
    .maxpool2d(2, 2, 2, 2)                       // pooling 2x2, stride 2x2, padding 0x0
    .flatten()                                    // Làm phẳng cho lớp dense
    .dense(128, "relu")                           // 128 đặc trưng đầu ra
    .dropout(0.5)                                 // 50% dropout 
    .dense(10, "linear")                          // 10 đặc trưng đầu ra, kích hoạt tuyến tính
    .build();

// Cấu hình huấn luyện
auto optimizer = std::make_unique<tnn::Adam<float>>(0.001f, 0.9f, 0.999f, 1e-8f);
model.set_optimizer(std::move(optimizer));

auto loss = tnn::LossFactory<float>::create_crossentropy(1e-15f);
model.set_loss_function(std::move(loss));

// Bật profiling (tùy chọn)
model.enable_profiling(true);

// Vòng lặp huấn luyện (đơn giản hóa)
for (int epoch = 0; epoch < so_epochs; ++epoch) {
    // ... mã huấn luyện như đã trình bày ở trên ...
}

// Lưu mô hình đã huấn luyện
model.save_to_file("mo_hinh_da_huan_luyen");
```

## Lợi Ích Chính

1. **Tự Động Suy Luận Hình Dạng**: Không cần tính toán thủ công kích thước giữa các lớp
2. **Giao Diện Linh Hoạt**: Nối các lớp thông qua method chaining
3. **An Toàn Kiểu**: Thiết kế dựa trên template đảm bảo tính nhất quán của kiểu
4. **Cấu Hình Linh Hoạt**: Hỗ trợ nhiều optimizer và hàm loss khác nhau
5. **Theo Dõi Hiệu Suất**: Khả năng profiling và debug tích hợp
6. **Lưu Trữ Mô Hình**: Dễ dàng lưu và tải mô hình hoàn chỉnh

Thiết kế này làm cho việc xây dựng và huấn luyện mạng neural trở nên trực quan hơn trong khi vẫn duy trì tính linh hoạt cần thiết cho các kiến trúc phức tạp.
