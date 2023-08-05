add_target_with_lib(test_binning test_binning.cpp)
add_target_with_lib(test_integrator test_integrator.cpp)

add_executable(test_vf test_vf.cpp)
target_link_libraries(test_vf PUBLIC Kokkos::kokkos)

add_executable(test test.cpp)
target_link_libraries(test PUBLIC Kokkos::kokkos)
