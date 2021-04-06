cmake_minimum_required(VERSION 3.17)

# definitions
set(PROJECT_NAME cppdrl)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/bin")
set(CMAKE_TOOLCHAIN_FILE ~/vcpkg/scripts/buildsystems/vcpkg.cmake CACHE STRING "Vcpkg toolchain file")

project(${PROJECT_NAME})
set(CMAKE_CXX_STANDARD 17)
add_compile_options( -larmadillo -lmlpack -fopenmp)
# Cppcheck
if (CMAKE_SOURCE_DIR STREQUAL CMAKE_CURRENT_SOURCE_DIR)
    list(APPEND CPPCHECK_ARGS
            --enable=warning
            --std=c++14
            --force
            --verbose
            --quiet
            --inline-suppr
            --error-exitcode=1
            --language=c++
            --config-exclude=${CMAKE_CURRENT_LIST_DIR}/src/third_party
            --config-exclude=${CMAKE_CURRENT_LIST_DIR}/lib
            -i${CMAKE_CURRENT_LIST_DIR}/example/lib
            --suppressions-list=${CMAKE_CURRENT_LIST_DIR}/CppCheckSuppressions.txt
            -I ${CMAKE_CURRENT_LIST_DIR}/src
            -I ${CMAKE_CURRENT_LIST_DIR}/include
            -I ${CMAKE_CURRENT_LIST_DIR}/example
            ${CMAKE_CURRENT_LIST_DIR}/src
            ${CMAKE_CURRENT_LIST_DIR}/example
            )

    add_custom_target(
            check
            COMMAND cppcheck ${CPPCHECK_ARGS}
            COMMENT "Running Cppcheck"
    )
endif(CMAKE_SOURCE_DIR STREQUAL CMAKE_CURRENT_SOURCE_DIR)

if (NOT TORCH_FOUND)
    find_package(Torch REQUIRED)
    if (TORCH_CXX_FLAGS)
        set(CMAKE_CXX_FLAGS ${TORCH_CXX_FLAGS})
    endif()
endif (NOT TORCH_FOUND)

find_package(mlpack CONFIG REQUIRED)

# dependencies
find_package(Boost COMPONENTS filesystem system unit_test_framework REQUIRED)
find_package(Protobuf REQUIRED )
find_package(OpenCV REQUIRED )
find_package(ale REQUIRED)
find_package(Catch2 CONFIG REQUIRED)

include_directories(include)
message(${OpenCV_INCLUDE_DIRS})
# Assign the include directories
include_directories(${Boost_INCLUDE_DIRS})


set(source_list
        ${CMAKE_CURRENT_LIST_DIR}/src/gym/spaces/box.cpp
        ${CMAKE_CURRENT_LIST_DIR}/src/gym/spaces/discrete.cpp
        ${CMAKE_CURRENT_LIST_DIR}/src/misc/helper.cpp
        ${CMAKE_CURRENT_LIST_DIR}/src/misc/strategy.cpp
        ${CMAKE_CURRENT_LIST_DIR}/src/models/mlp.cpp)

set(include_list  include/cppdrl/cppdrl.h)

add_subdirectory(Catch_tests)

set(SOURCE_FILES main.cpp ${TORCH_INCLUDE_DIRS} ${source_list} ${include_list} include/cppdrl/gym/cartpole.h)

add_executable(${PROJECT_NAME} ${SOURCE_FILES})

target_link_libraries(${PROJECT_NAME}  ${TORCH_LIBRARIES} ${OpenCV_LIBS} ale::ale-lib ${Boost_LIBRARIES} mlpack::mlpack  Catch2::Catch2)

