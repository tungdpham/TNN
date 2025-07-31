cmake_minimum_required(VERSION 3.10)
project(FedCNN)

# Set the C++ standard to 17
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Add the executable for the CIFAR-100 CNN trainer
add_executable(cifar100_cnn_trainer cifar100_cnn_trainer.cpp)
target_link_libraries(cifar100_cnn_trainer PRIVATE blas)

# Add the executable for the CIFAR-10 CNN trainer
add_executable(cifar10_cnn_trainer cifar10_cnn_trainer.cpp)
target_link_libraries(cifar10_cnn_trainer PRIVATE blas)