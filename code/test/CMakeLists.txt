add_target_with_lib(test_binning test_binning.cpp)
add_executable(test test.cpp)
target_link_libraries(test PUBLIC Kokkos::kokkos)

