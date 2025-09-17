# DCNN Library Components

add_library(dcnn_matrix STATIC matrix.cpp)
if(NOT MSVC)
    target_link_libraries(dcnn_matrix PUBLIC m)
endif()

add_library(dcnn_tensor STATIC tensor.cpp)
if(NOT MSVC)
    target_link_libraries(dcnn_tensor PUBLIC dcnn_matrix m)
else()
    target_link_libraries(dcnn_tensor PUBLIC dcnn_matrix)
endif()

# Check for optional libraries
find_package(PkgConfig QUIET)
if(PkgConfig_FOUND)
    pkg_check_modules(NUMA numa)
endif()

add_library(dcnn_utils STATIC utils.cpp hardware_info.cpp)

# Add assembly optimizations for x86_64
if(CMAKE_SYSTEM_PROCESSOR MATCHES "x86_64|AMD64")
    if(WIN32 AND MSVC AND NOT DISABLE_ASM)
        target_sources(dcnn_utils PRIVATE ${CMAKE_SOURCE_DIR}/include/utils/simd_dot_product_win.asm)
        set_property(SOURCE ${CMAKE_SOURCE_DIR}/include/utils/simd_dot_product_win.asm PROPERTY LANGUAGE ASM_NASM)
        message(STATUS "Using NASM assembly optimizations")
    elseif(NOT WIN32 OR NOT MSVC)
        target_sources(dcnn_utils PRIVATE ${CMAKE_SOURCE_DIR}/include/utils/simd_dot_product.S)
        set_property(SOURCE ${CMAKE_SOURCE_DIR}/include/utils/simd_dot_product.S PROPERTY LANGUAGE ASM)
        message(STATUS "Using GNU assembly optimizations")
    else()
        message(STATUS "Assembly optimizations disabled")
    endif()
endif()

if(NOT MSVC)
    target_link_libraries(dcnn_utils PUBLIC m pthread)
    if(WIN32 AND MINGW)
        target_link_libraries(dcnn_utils PUBLIC pdh)
    endif()
    if(NUMA_FOUND)
        target_link_libraries(dcnn_utils PUBLIC ${NUMA_LIBRARIES})
        target_include_directories(dcnn_utils PUBLIC ${NUMA_INCLUDE_DIRS})
        target_compile_definitions(dcnn_utils PUBLIC NUMA_VERSION1_COMPATIBILITY)
    endif()
else()
    target_link_libraries(dcnn_utils PUBLIC pdh)
endif()

add_library(dcnn_data_loading STATIC data_loading.cpp)
target_link_libraries(dcnn_data_loading PUBLIC dcnn_utils dcnn_tensor dcnn_matrix)

add_library(dcnn_nn STATIC nn.cpp)
if(NOT MSVC)
    target_link_libraries(dcnn_nn PUBLIC dcnn_tensor dcnn_matrix dcnn_utils dcnn_data_loading m)
else()
    target_link_libraries(dcnn_nn PUBLIC dcnn_tensor dcnn_matrix dcnn_utils dcnn_data_loading)
endif()
target_link_libraries(dcnn_nn PUBLIC nlohmann_json::nlohmann_json)

add_library(dcnn_pipeline STATIC pipeline.cpp)
target_link_libraries(dcnn_pipeline PUBLIC dcnn_nn protobuf::libprotobuf)
target_include_directories(dcnn_pipeline PUBLIC ${CMAKE_CURRENT_BINARY_DIR})
target_link_libraries(dcnn_pipeline PUBLIC nlohmann_json::nlohmann_json)
target_link_libraries(dcnn_pipeline PUBLIC libzstd_static)
target_include_directories(dcnn_pipeline PUBLIC ${asio_SOURCE_DIR}/asio/include)
target_compile_definitions(dcnn_pipeline PUBLIC ASIO_STANDALONE)

set(PROTO_DIR "${CMAKE_SOURCE_DIR}/include/pipeline/proto")
# auto generate proto files
if(EXISTS "${PROTO_DIR}")
    file(GLOB PROTO_FILES "${PROTO_DIR}/*.proto")
    if(PROTO_FILES)
        set(PROTO_OUT_DIR "${CMAKE_CURRENT_BINARY_DIR}/generated")
        file(MAKE_DIRECTORY ${PROTO_OUT_DIR})
        
        protobuf_generate_cpp(PROTO_SRCS PROTO_HDRS ${PROTO_FILES})
        
        set_source_files_properties(${PROTO_SRCS} ${PROTO_HDRS} PROPERTIES
            GENERATED TRUE
        )
        
        add_library(dcnn_proto STATIC ${PROTO_SRCS})
        target_link_libraries(dcnn_proto PUBLIC protobuf::libprotobuf)
        target_include_directories(dcnn_proto PUBLIC 
            ${CMAKE_CURRENT_BINARY_DIR}
            ${CMAKE_SOURCE_DIR}/include
        )
        
        target_link_libraries(dcnn_pipeline PUBLIC dcnn_proto)
        
        target_include_directories(dcnn_pipeline PUBLIC ${CMAKE_CURRENT_BINARY_DIR})
    endif()
endif()

add_library(dcnn_instantiations INTERFACE)
target_link_libraries(dcnn_instantiations INTERFACE dcnn_pipeline dcnn_nn dcnn_data_loading dcnn_tensor dcnn_matrix dcnn_utils)

add_library(dcnn_lib INTERFACE)
if(NOT MSVC)
    target_link_libraries(dcnn_lib INTERFACE dcnn_instantiations m pthread)
else()
    target_link_libraries(dcnn_lib INTERFACE dcnn_instantiations)
endif()

target_link_libraries(dcnn_lib INTERFACE nlohmann_json::nlohmann_json)
target_link_libraries(dcnn_lib INTERFACE libzstd_static)
target_include_directories(dcnn_lib INTERFACE ${asio_SOURCE_DIR}/asio/include)
target_compile_definitions(dcnn_lib INTERFACE ASIO_STANDALONE)

# Add Windows networking libraries for ASIO support
if(WIN32)
    target_link_libraries(dcnn_lib INTERFACE ws2_32 wsock32 mswsock)
    if(MINGW)
        target_link_libraries(dcnn_lib INTERFACE iphlpapi)
    endif()
endif()

set(INSTANTIATION_LIBS dcnn_matrix dcnn_tensor dcnn_nn dcnn_utils)

if(ENABLE_OPENMP)
    foreach(lib ${INSTANTIATION_LIBS})
        target_link_libraries(${lib} PUBLIC OpenMP::OpenMP_CXX)
    endforeach()
    target_link_libraries(dcnn_lib INTERFACE OpenMP::OpenMP_CXX)
endif()

if(ENABLE_TBB)
    foreach(lib ${INSTANTIATION_LIBS})
        target_link_libraries(${lib} PUBLIC TBB::tbb)
    endforeach()
    target_link_libraries(dcnn_lib INTERFACE TBB::tbb)
endif()

if(ENABLE_CUDA)
    foreach(lib ${INSTANTIATION_LIBS})
        target_link_libraries(${lib} PUBLIC CUDA::cudart CUDA::cublas CUDA::curand)
    endforeach()
    target_link_libraries(dcnn_lib INTERFACE CUDA::cudart CUDA::cublas CUDA::curand)
endif()