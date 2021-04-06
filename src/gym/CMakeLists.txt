target_sources(cppdrl
        PRIVATE
        ${CMAKE_CURRENT_LIST_DIR}/base_atari.cpp
        ${CMAKE_CURRENT_LIST_DIR}/base_wrappers.cpp
        )

add_subdirectory(spaces)
add_subdirectory(wrappers)