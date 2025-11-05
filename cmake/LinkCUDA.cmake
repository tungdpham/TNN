function(link_cuda target_name)
  if(ENABLE_CUDA)
      set_target_properties(${target_name} PROPERTIES 
          CUDA_ARCHITECTURES "89"
          CUDA_STANDARD 17
      )
      target_link_libraries(${target_name} CUDA::cudart)
  endif()
endfunction()
