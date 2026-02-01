# Unit Tests CMakeLists.txt

include(GoogleTest)

set(TEST_FILES 
    tensor_test.cpp
    slice_ops_test.cpp
    device_manager_test.cpp
    cuda_kernels_test.cpp
    gpu_tensor_test.cpp
    gpu_tensor_ops_test.cpp
    cuda_conv2d_nchw_ops_test.cpp
    cuda_dense_ops_test.cpp
    cuda_maxpool_ops_test.cpp
    cuda_loss_funcs_test.cpp
    cuda_batchnorm_ops_test.cpp
    layer_device_agnosticity_test.cpp
    layer_buffer_reuse_test.cpp
    residual_block_test.cpp
    sequential_residual_block_test.cpp
    tiny_imagenet_loader_test.cpp
    legacy_dense_layer_test.cpp
    legacy_maxpool2d_layer_test.cpp
    legacy_avgpool_layer_test.cpp
    legacy_batchnorm_layer_test.cpp
    legacy_conv2d_layer_test.cpp
    groupnorm_layer_test.cpp
    attention_block_test.cpp
    vbuffer_test.cpp
    vserializer_test.cpp
)

foreach(test_file IN LISTS TEST_FILES)
    get_filename_component(test_name ${test_file} NAME_WE)
    create_executable(${test_file} LIBS tnn_lib gtest_main gtest gmock)
    gtest_discover_tests(${test_name} DISCOVERY_MODE PRE_TEST)
endforeach()