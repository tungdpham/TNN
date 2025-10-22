# Utility functions for the DCNN project

# Function to create executables with common linking configuration
# Usage: create_executable(target_name source_file)
function(create_executable target_name source_file)
    add_executable(${target_name} ${source_file})
    
    if(ENABLE_CUDA)
        set_property(TARGET ${target_name} PROPERTY CUDA_ARCHITECTURES 89)
    endif()
    
    # Link core DCNN libraries
    target_link_libraries(${target_name} PRIVATE 
        dcnn_lib 
        dcnn_pipeline 
        dcnn_nn 
        dcnn_data_loading 
        dcnn_math 
        dcnn_tensor 
        dcnn_matrix 
        dcnn_utils
    )
    
    # Link third-party dependencies
    target_link_libraries(${target_name} PRIVATE nlohmann_json::nlohmann_json)
    target_link_libraries(${target_name} PRIVATE libzstd_static)
    
    # ASIO configuration
    target_include_directories(${target_name} PRIVATE ${asio_SOURCE_DIR}/asio/include)
    target_compile_definitions(${target_name} PRIVATE ASIO_STANDALONE)
    
    # Optional dependencies
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
    
    # Windows-specific libraries
    if(WIN32)
        target_link_libraries(${target_name} PRIVATE ${WINDOWS_LIBS})
    endif()
endfunction()
