# Create an executable with proper linking and configurations
function(create_executable source_file)
    # Derive target name from source file (remove extension)
    get_filename_component(target_name ${source_file} NAME_WE)
    
    cmake_parse_arguments(
        ARG
        "USE_CUDA"                   
        "LINK_MODE"
        "LIBS"
        ${ARGN}
    )
    
    add_executable(${target_name} ${source_file})
    
    if(NOT ARG_LINK_MODE)
        set(ARG_LINK_MODE PRIVATE)
    endif()
    
    if(ARG_LIBS)
        target_link_libraries(${target_name} ${ARG_LINK_MODE} ${ARG_LIBS})
    endif()
    
    if(ARG_USE_CUDA AND ENABLE_CUDA)
        target_link_libraries(${target_name} PRIVATE CUDA::cudart)
    endif()
endfunction()