# Training executables
create_default(mnist_cnn_trainer mnist_cnn_trainer.cpp)

create_default(cifar10_cnn_trainer cifar10_cnn_trainer.cpp)
create_default(cifar100_cnn_trainer cifar100_cnn_trainer.cpp)

create_default(tiny_imagenet_resnet18 tiny_imagenet_resnet18.cpp)
create_default(tiny_imagenet_resnet34 tiny_imagenet_resnet34.cpp)
create_default(tiny_imagenet_resnet50 tiny_imagenet_resnet50.cpp)
create_default(tiny_imagenet_vit_trainer tiny_imagenet_vit_trainer.cpp)
create_default(gpt2_trainer gpt2_trainer.cpp)
create_default(gpt2_inference gpt2_inference.cpp)

create_default(cifar10_resnet9 cifar10_resnet9.cpp)

# Examples
create_default(hardware_info_example hardware_info_example.cpp)
create_default(device_manager_example device_manager_example.cpp)
create_cuda(device_ptr_example device_ptr_example.cu)

# Eval executables
create_default(mnist_cnn_test mnist_cnn_test.cpp)

# Network executables
create_default(tcp_worker tcp_worker.cpp)
create_default(tcp_coordinator tcp_coordinator.cpp)
create_default(coordinator_tiny_imagenet coordinator_tiny_imagenet.cpp)

# Microbatching test executable
create_default(roce_worker roce_worker.cpp)
create_default(roce_coordinator roce_coordinator.cpp)
create_default(inspect_openwebtext inspect_openwebtext.cpp)
