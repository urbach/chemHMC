
add_library(
    particles STATIC
    particles.hpp
    ../modules/particles.cpp
) 

target_link_libraries(particles PUBLIC ${YAML_CPP_LIBRARIES} read_infile Kokkos::kokkos)
target_include_directories(particles PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}")

add_library(
    read_infile STATIC
    read_infile.hpp
    ../modules/read_infile.cpp
) 
set_target_properties(read_infile PROPERTIES LINKER_LANGUAGE CXX)
target_link_libraries(read_infile PUBLIC ${YAML_CPP_LIBRARIES})
target_link_libraries(read_infile PUBLIC particles)
target_include_directories(read_infile PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}")


add_library(
    HMC STATIC
    HMC.hpp
    ../modules/HMC.cpp
) 

target_link_libraries(HMC PUBLIC ${YAML_CPP_LIBRARIES} read_infile Kokkos::kokkos)
target_include_directories(HMC PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}")



add_library(
    integrator STATIC
    integrator.hpp
    ../modules/integrator.cpp
) 

target_link_libraries(integrator PUBLIC ${YAML_CPP_LIBRARIES} read_infile Kokkos::kokkos)
target_include_directories(integrator PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}")
