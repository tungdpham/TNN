# TNN CMakeLists.txt

# CUDA library
if(ENABLE_CUDA)
    file(GLOB_RECURSE TNN_CUDA_SOURCES 
        "cuda/*.cu"
    )
    add_library(tnn_cuda STATIC ${TNN_CUDA_SOURCES})
    target_include_directories(tnn_cuda PUBLIC 
        $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/include>
        $<INSTALL_INTERFACE:include>
    )
    set_target_properties(tnn_cuda PROPERTIES 
        CUDA_ARCHITECTURES ${CMAKE_CUDA_ARCHITECTURES}
        CUDA_STANDARD 17
    )
    target_link_libraries(tnn_cuda PUBLIC CUDA::cudart CUDA::cublas)
endif()

# Device library
file(GLOB_RECURSE TNN_DEVICE_SOURCES 
    "device/*.cpp"
)
add_library(tnn_device STATIC ${TNN_DEVICE_SOURCES})
target_include_directories(tnn_device PUBLIC 
    $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>
)
if(ENABLE_CUDA)
    target_link_libraries(tnn_device PUBLIC CUDA::cudart CUDA::cublas)
endif()

# Operations library
file(GLOB_RECURSE TNN_OPS_SOURCES 
    "ops/*.cpp"
    "ops/*.cu"
)
add_library(tnn_ops STATIC ${TNN_OPS_SOURCES})
target_include_directories(tnn_ops PUBLIC 
    $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>
)
target_link_libraries(tnn_ops PUBLIC tnn_device)
if(ENABLE_CUDA)
    target_link_libraries(tnn_ops PUBLIC CUDA::cudart CUDA::cublas)
endif()

# Matrix library
file(GLOB_RECURSE TNN_MATRIX_SOURCES 
    "matrix/*.cpp"
)
add_library(tnn_matrix STATIC ${TNN_MATRIX_SOURCES})
target_include_directories(tnn_matrix PUBLIC 
    $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>
)
target_link_libraries(tnn_matrix PUBLIC tnn_device tnn_ops)
if(NOT MSVC)
    target_link_libraries(tnn_matrix PUBLIC m)
endif()

# Tensor library
file(GLOB_RECURSE TNN_TENSOR_SOURCES 
    "tensor/*.cpp"
)
add_library(tnn_tensor STATIC ${TNN_TENSOR_SOURCES})
target_include_directories(tnn_tensor PUBLIC 
    $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>
)
target_link_libraries(tnn_tensor PUBLIC tnn_matrix tnn_ops tnn_device)
if(NOT MSVC)
    target_link_libraries(tnn_tensor PUBLIC m)
endif()

# Utilities library
file(GLOB_RECURSE TNN_UTILS_SOURCES 
    "utils/*.cpp"
)
add_library(tnn_utils STATIC ${TNN_UTILS_SOURCES})
target_include_directories(tnn_utils PUBLIC 
    $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>
)

# add assembly optimizations for x86_64
if(CMAKE_SYSTEM_PROCESSOR MATCHES "x86_64|AMD64")
    if(WIN32 AND MSVC AND NOT DISABLE_ASM)
        file(GLOB ASM_SOURCES "utils/*.asm")
        target_sources(tnn_utils PRIVATE ${ASM_SOURCES})
        foreach(asm_file ${ASM_SOURCES})
            set_property(SOURCE ${asm_file} PROPERTY LANGUAGE ASM_NASM)
        endforeach()
        message(STATUS "Using NASM assembly optimizations")
    elseif(NOT WIN32 OR NOT MSVC)
        file(GLOB ASM_SOURCES "utils/*.S")
        target_sources(tnn_utils PRIVATE ${ASM_SOURCES})
        foreach(asm_file ${ASM_SOURCES})
            set_property(SOURCE ${asm_file} PROPERTY LANGUAGE ASM)
        endforeach()
        message(STATUS "Using GNU assembly optimizations")
    else()
        message(STATUS "Assembly optimizations disabled")
    endif()
endif()

# add platform-specific lib for hardware info
if(NOT MSVC)
    target_link_libraries(tnn_utils PUBLIC m pthread)
    if(WIN32 AND MINGW)
        target_link_libraries(tnn_utils PUBLIC pdh)
    endif()
    if(NUMA_FOUND)
        target_link_libraries(tnn_utils PUBLIC ${NUMA_LIBRARIES})
        target_include_directories(tnn_utils PUBLIC ${NUMA_INCLUDE_DIRS})
        target_compile_definitions(tnn_utils PUBLIC NUMA_VERSION1_COMPATIBILITY)
    endif()
else()
    target_link_libraries(tnn_utils PUBLIC pdh)
endif()

# Data loading library
file(GLOB_RECURSE TNN_DATA_LOADING_SOURCES 
    "data_loading/*.cpp"
)
add_library(tnn_data_loading STATIC ${TNN_DATA_LOADING_SOURCES})
target_include_directories(tnn_data_loading PUBLIC 
    $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>
)
target_link_libraries(tnn_data_loading PUBLIC tnn_utils tnn_tensor tnn_matrix)

# Data augmentation library
file(GLOB_RECURSE TNN_DATA_AUGMENTATION_SOURCES
    "data_augmentation/*.cpp"
)
add_library(tnn_data_augmentation STATIC ${TNN_DATA_AUGMENTATION_SOURCES})
target_include_directories(tnn_data_augmentation PUBLIC 
    $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>
)
target_link_libraries(tnn_data_augmentation PUBLIC tnn_tensor)

# Math library
file(GLOB_RECURSE TNN_MATH_SOURCES 
    "math/*.cpp"
)
if(ENABLE_CUDA)
    file(GLOB_RECURSE TNN_MATH_CUDA_SOURCES 
        "math/*.cu"
    )
    add_library(tnn_math STATIC ${TNN_MATH_SOURCES} ${TNN_MATH_CUDA_SOURCES})
else()
    add_library(tnn_math STATIC ${TNN_MATH_SOURCES})
endif()
target_include_directories(tnn_math PUBLIC 
    $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>
)
target_link_libraries(tnn_math PUBLIC tnn_utils)
if(ENABLE_CUDA)
    set_target_properties(tnn_math PROPERTIES 
        CUDA_ARCHITECTURES ${CMAKE_CUDA_ARCHITECTURES}
        CUDA_STANDARD 17
    )
    target_link_libraries(tnn_math PUBLIC CUDA::cudart CUDA::cublas)
endif()

# Neural Network library
file(GLOB_RECURSE TNN_NN_SOURCES 
    "nn/*.cpp"
)
file(GLOB_RECURSE TNN_NN_CUDA_SOURCES 
    "nn/*.cu"
)
add_library(tnn_nn STATIC ${TNN_NN_SOURCES} ${TNN_NN_CUDA_SOURCES})
target_include_directories(tnn_nn PUBLIC 
    $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>
)
target_link_libraries(tnn_nn PUBLIC tnn_tensor tnn_utils tnn_data_loading tnn_math tnn_device)
target_link_libraries(tnn_nn PUBLIC nlohmann_json::nlohmann_json)
if(ENABLE_CUDA)
    target_link_libraries(tnn_nn PUBLIC CUDA::cudart CUDA::cublas)
endif()

# Partitioner module
file(GLOB_RECURSE TNN_PARTITIONER_SOURCES 
    "partitioner/*.cpp"
)
add_library(tnn_partitioner STATIC ${TNN_PARTITIONER_SOURCES})
target_include_directories(tnn_partitioner PUBLIC 
    $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>
)
target_link_libraries(tnn_partitioner PUBLIC tnn_nn)

# Pipeline library
file(GLOB_RECURSE TNN_PIPELINE_SOURCES 
    "pipeline/*.cpp"
)
add_library(tnn_pipeline STATIC ${TNN_PIPELINE_SOURCES})
target_include_directories(tnn_pipeline PUBLIC 
    $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/include>
    $<BUILD_INTERFACE:${CMAKE_CURRENT_BINARY_DIR}>
    $<INSTALL_INTERFACE:include>
)
# Private include for build-time dependencies
target_include_directories(tnn_pipeline PRIVATE 
    ${asio_SOURCE_DIR}/asio/include
)
target_link_libraries(tnn_pipeline PUBLIC nlohmann_json::nlohmann_json)
target_link_libraries(tnn_pipeline PUBLIC libzstd_static)
target_compile_definitions(tnn_pipeline PUBLIC ASIO_STANDALONE)

# Main TNN library
set(TNN_MODULES tnn_matrix tnn_tensor tnn_nn tnn_utils tnn_math tnn_data_loading tnn_pipeline tnn_device)
if(ENABLE_CUDA)
    list(APPEND TNN_MODULES tnn_cuda)
endif()

add_library(tnn_lib INTERFACE)
target_include_directories(tnn_lib INTERFACE
    $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>
)
if(NOT MSVC)
    target_link_libraries(tnn_lib INTERFACE ${TNN_MODULES} m pthread)
else()
    target_link_libraries(tnn_lib INTERFACE ${TNN_MODULES})
endif()

if(WIN32)
    target_link_libraries(tnn_lib INTERFACE ws2_32 wsock32 mswsock)
    if(MINGW)
        target_link_libraries(tnn_lib INTERFACE iphlpapi)
    endif()
endif()


foreach(lib ${TNN_MODULES})
    link_tbb(PUBLIC ${lib})
    link_cuda(PUBLIC ${lib})
    link_mkl(PUBLIC ${lib})
    link_openmp(PUBLIC ${lib})
endforeach()

link_tbb(INTERFACE tnn_lib)
link_cuda(INTERFACE tnn_lib)
link_mkl(INTERFACE tnn_lib)
link_openmp(INTERFACE tnn_lib)