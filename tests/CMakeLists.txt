

# Project settings
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ".")
set(PROJECT_UNIT_TESTS_NAME cppdrl_unit_tests)

set (Boost_USE_STATIC_LIBS OFF)
# Dependencies
find_package(Boost COMPONENTS filesystem system unit_test_framework REQUIRED)
#find_package(mlpack CONFIG REQUIRED)
#add_compile_options( -larmadillo -lmlpack -fopenmp)
# Include paths
set(UNIT_TESTS_INCLUDES ${PROJECT_INCLUDE_DIR}
        ../include/cppdrl
        ${Boost_INCLUDE_DIR})

include_directories (${Boost_INCLUDE_DIRS})
include_directories(${UNIT_TESTS_INCLUDES})
add_definitions(-DBOOST_TEST_DYN_LINK)

add_executable(${PROJECT_UNIT_TESTS_NAME} ${source_list}
#        test_space.cpp
#        test_envs.cpp
#        test_helper.cpp
        test_dqn.cpp)
target_link_libraries(${PROJECT_UNIT_TESTS_NAME} ${TORCH_LIBRARIES} ${OpenCV_LIBS} ale::ale-lib ${Boost_LIBRARIES} mlpack::mlpack  Catch2::Catch2)
