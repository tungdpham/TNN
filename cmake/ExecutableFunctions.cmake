# Create an executable with proper linking and configurations
function(create_default target_name source_file)
    add_executable(${target_name} ${source_file})
    
    # Link core TNN libraries
    target_link_libraries(${target_name} PRIVATE 
        tnn_lib
    )
endfunction()

function(create_cuda target_name source_file)
    add_executable(${target_name} ${source_file})
    
    # Link core TNN libraries
    target_link_libraries(${target_name} PRIVATE 
        tnn_lib
        CUDA::cudart
    )
endfunction()
