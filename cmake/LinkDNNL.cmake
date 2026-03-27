function(link_dnnl visibility target_name)
    if(ENABLE_DNNL)
        if(TARGET DNNL::dnnl)
            target_link_libraries(${target_name} ${visibility} DNNL::dnnl)
        else()
            target_include_directories(${target_name} ${visibility} ${DNNL_INCLUDE_DIRS})
            target_link_libraries(${target_name} ${visibility} ${DNNL_LIBRARIES})
            if(NOT MSVC)
                target_link_libraries(${target_name} ${visibility} pthread m dl)
            endif()
        endif()
        # TBB-backed DNNL requires TBB at link time
        if(TARGET TBB::tbb)
            target_link_libraries(${target_name} ${visibility} TBB::tbb)
        endif()

        if(DNNL_INTEL_RUNTIME_LIB_DIR)
            target_link_directories(${target_name} ${visibility} "${DNNL_INTEL_RUNTIME_LIB_DIR}")
            if(NOT MSVC)
                # Use --disable-new-dtags so the linker embeds DT_RPATH instead of
                # DT_RUNPATH.  DT_RPATH is searched transitively (i.e. also for
                # libraries needed by libdnnl.so itself such as libsvml/libimf/libsycl),
                # whereas DT_RUNPATH is only searched for the binary's direct dependencies.
                target_link_options(${target_name} ${visibility}
                    "LINKER:--disable-new-dtags"
                    "LINKER:-rpath,${DNNL_INTEL_RUNTIME_LIB_DIR}")
            endif()
        endif()
        if(DNNL_INTEL_RUNTIME_LIBS)
            target_link_libraries(${target_name} ${visibility} ${DNNL_INTEL_RUNTIME_LIBS})
        endif()
    endif()
endfunction()
