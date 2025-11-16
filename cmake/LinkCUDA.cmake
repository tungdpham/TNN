function(link_cuda visibility target_name)
  if(ENABLE_CUDA)
      # Use the auto-detected architecture or fallback
      if(DEFINED CUDA_ARCH_NUMBER)
          set(TARGET_CUDA_ARCH ${CUDA_ARCH_NUMBER})
      else()
          set(TARGET_CUDA_ARCH "75") # Fallback
      endif()
      
      set_target_properties(${target_name} PROPERTIES 
          CUDA_ARCHITECTURES "${TARGET_CUDA_ARCH}"
          CUDA_STANDARD 17
      )
      target_link_libraries(${target_name} ${visibility} CUDA::cudart CUDA::cublas)
  endif()
endfunction()
