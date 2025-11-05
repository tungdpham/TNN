# Training executables
create_executable(mnist_cnn_trainer mnist_cnn_trainer.cpp)
create_executable(cifar10_cnn_trainer cifar10_cnn_trainer.cpp)
create_executable(cifar10_cnn_trainer_v2 cifar10_cnn_trainer_v2.cpp)
create_executable(cifar100_cnn_trainer cifar100_cnn_trainer.cpp)
create_executable(uji_ips_trainer uji_ips_trainer.cpp)

# Examples
create_executable(hardware_info_example hardware_info_example.cpp)
create_executable(device_manager_example device_manager_example.cpp)

# CUDA examples (only built if CUDA is enabled)
if(ENABLE_CUDA)
    add_executable(simple_cuda_test simple_cuda_test.cu)
    
    # Set CUDA properties
    set_target_properties(simple_cuda_test PROPERTIES 
        CUDA_ARCHITECTURES "89"
        CUDA_STANDARD 17
    )
    
    # Link libraries
    target_link_libraries(simple_cuda_test PRIVATE 
        tnn_device
        tnn_utils
        CUDA::cudart
        nlohmann_json::nlohmann_json
    )
    
    # Include directories
    target_include_directories(simple_cuda_test PRIVATE 
        ${CMAKE_SOURCE_DIR}/include
        ${asio_SOURCE_DIR}/asio/include
    )
    
    # Compile definitions
    target_compile_definitions(simple_cuda_test PRIVATE 
        USE_CUDA
        ASIO_STANDALONE
    )
endif()

# Eval executables
create_executable(mnist_cnn_test mnist_cnn_test.cpp)

# Network executables
create_executable(network_worker network_worker.cpp)
create_executable(sync_pipeline_coordinator sync_pipeline_coordinator.cpp)
create_executable(semi_async_pipeline_coordinator semi_async_pipeline_coordinator.cpp)