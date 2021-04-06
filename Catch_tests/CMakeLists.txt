
find_package(Catch2 CONFIG REQUIRED)
add_executable(Catch_tests_run test_dqn.cpp  ${TORCH_INCLUDE_DIRS} ${source_list} )

target_link_libraries(Catch_tests_run  ${TORCH_LIBRARIES} ${OpenCV_LIBS} ale::ale-lib ${Boost_LIBRARIES} mlpack::mlpack)