function(create_math_test target_name source_file)
    add_executable(${target_name} ${source_file})
    if(ENABLE_CUDA)
        set_property(TARGET ${target_name} PROPERTY CUDA_ARCHITECTURES 89)
    endif()
    
    target_link_libraries(${target_name} PRIVATE dcnn_data_loading dcnn_math dcnn_tensor dcnn_matrix dcnn_utils)
    
    target_link_libraries(${target_name} PRIVATE nlohmann_json::nlohmann_json)
    
    if(ENABLE_TBB)
        target_link_libraries(${target_name} PRIVATE TBB::tbb)
    endif()
    
    if(ENABLE_MKL)
        if(TARGET MKL::MKL)
            target_link_libraries(${target_name} PRIVATE MKL::MKL)
        else()
            target_include_directories(${target_name} PRIVATE ${MKL_INCLUDE_DIRS})
            target_link_libraries(${target_name} PRIVATE ${MKL_LIBRARIES})
            
            if(NOT MSVC)
                target_link_libraries(${target_name} PRIVATE pthread m dl)
            endif()

        endif()
    endif()

    
    if(WIN32)
        target_link_libraries(${target_name} PRIVATE ${WINDOWS_LIBS})
    endif()
endfunction()

create_math_test(serialization_test serialization_test.cpp)
create_math_test(compression_test compression_test.cpp)
create_math_test(resize_test resize_test.cpp)
create_math_test(reserve_test reserve_test.cpp)
create_math_test(gemm_cache gemm_cache.cpp)
create_math_test(gemm_test gemm_test.cpp)