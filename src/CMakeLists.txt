# DCNN Library Components

# Create instantiation libraries in dependency order

add_library(dcnn_matrix STATIC matrix.cpp)
target_link_libraries(dcnn_matrix PUBLIC m)

add_library(dcnn_tensor STATIC tensor.cpp)
target_link_libraries(dcnn_tensor PUBLIC dcnn_matrix m)

add_library(dcnn_nn STATIC nn.cpp)
target_link_libraries(dcnn_nn PUBLIC dcnn_tensor dcnn_matrix m)

add_library(dcnn_instantiations INTERFACE)
target_link_libraries(dcnn_instantiations INTERFACE dcnn_nn dcnn_tensor dcnn_matrix)

add_library(dcnn_lib INTERFACE)
target_link_libraries(dcnn_lib INTERFACE dcnn_instantiations m)

set(INSTANTIATION_LIBS dcnn_matrix dcnn_tensor dcnn_nn)

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
add_executable(cpu_logger cpu_logger.cpp)
target_compile_features(cpu_logger PRIVATE cxx_std_17)
target_link_libraries(cpu_logger PRIVATE pthread)
