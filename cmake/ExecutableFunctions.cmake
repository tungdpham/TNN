# Create an executable with proper linking and configurations
function(create_executable target_name source_file)
    add_executable(${target_name} ${source_file})
    
    # Link core TNN libraries
    target_link_libraries(${target_name} PRIVATE 
        tnn_lib
    )
    
    link_tbb(${target_name})
    link_mkl(${target_name})
    link_cuda(${target_name})
    link_windows_libs(${target_name})
endfunction()
