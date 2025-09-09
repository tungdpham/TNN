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

add_library(dcnn_utils STATIC utils.cpp cpu_info.cpp)

if(CMAKE_SYSTEM_PROCESSOR MATCHES "x86_64|AMD64")
    target_sources(dcnn_utils PRIVATE ${CMAKE_SOURCE_DIR}/include/utils/simd_dot_product.S)
    set_property(SOURCE ${CMAKE_SOURCE_DIR}/include/utils/simd_dot_product.S PROPERTY LANGUAGE ASM)
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

add_library(dcnn_nn STATIC nn.cpp)
if(NOT MSVC)
    target_link_libraries(dcnn_nn PUBLIC dcnn_tensor dcnn_matrix dcnn_utils m)
else()
    target_link_libraries(dcnn_nn PUBLIC dcnn_tensor dcnn_matrix dcnn_utils)
endif()

add_library(dcnn_pipeline STATIC pipeline.cpp pipeline.pb.cc)
target_link_libraries(dcnn_pipeline PUBLIC dcnn_nn ${Protobuf_LIBRARIES})
target_include_directories(dcnn_pipeline PUBLIC ${Protobuf_INCLUDE_DIRS})

add_library(dcnn_instantiations INTERFACE)
target_link_libraries(dcnn_instantiations INTERFACE dcnn_pipeline dcnn_nn dcnn_tensor dcnn_matrix dcnn_utils)

add_library(dcnn_lib INTERFACE)
if(NOT MSVC)
    target_link_libraries(dcnn_lib INTERFACE dcnn_instantiations m pthread)
else()
    target_link_libraries(dcnn_lib INTERFACE dcnn_instantiations)
endif()

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

# Build cpu_logger utility
if(NOT WIN32) 
    add_executable(cpu_logger cpu_logger.cpp)
    target_compile_features(cpu_logger PRIVATE cxx_std_17)
    target_link_libraries(cpu_logger PRIVATE pthread)
endif()