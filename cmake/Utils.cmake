# Create an executable with proper linking and configurations
function(create_executable target_name source_file)
    add_executable(${target_name} ${source_file})
    
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
    
    link_tbb(${target_name})
    link_mkl(${target_name})
    link_cuda(${target_name})
    link_windows_libs(${target_name})
endfunction()
