find_package(TBB CONFIG REQUIRED)

set(TBB_TARGETS TBB::tbb TBB::tbbmalloc)

foreach(target IN LISTS TBB_TARGETS)
    if(TARGET ${target})
        set_property(TARGET ${target} PROPERTY MAP_IMPORTED_CONFIG_DEBUG Release)
    endif()
endforeach()

add_compile_definitions(USE_TBB)