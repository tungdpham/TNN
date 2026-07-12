# Unit Tests CMakeLists.txt

include(GoogleTest)

set(TEST_FILES 
    test_tensor.cpp
    test_device_manager.cpp
    test_cuda_kernels.cpp
    test_gpu_tensor.cpp
    test_gpu_tensor_ops.cpp
    test_cuda_loss_funcs.cpp
    test_residual_block.cpp
    test_partitioner.cpp
    test_archiver.cpp
    test_bf16.cpp
    test_fp16.cpp
    test_cpu_engine.cpp
    test_cuda_engine.cpp
    test_cudnn_engine.cpp
)

foreach(test_file IN LISTS TEST_FILES)
    get_filename_component(test_name ${test_file} NAME_WE)
    if(ENABLE_CUDA)
        create_executable(${test_file} LIBS tunx_lib gtest_main gtest gmock CUDA::cudart)
    else()
        create_executable(${test_file} LIBS tunx_lib gtest_main gtest gmock)
    endif()
    gtest_discover_tests(${test_name} DISCOVERY_MODE PRE_TEST)
endforeach()