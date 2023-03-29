
set(chemHMC_header
    particles.hpp
    identical_particles.hpp
    read_infile.hpp
    HMC.hpp
    integrator.hpp
    # quick_sort.hpp
)

set(chemHMC_scr
    ../modules/particles.cpp
    ../modules/identical_particles.cpp
    ../modules/identical_particles_binning.cpp
    ../modules/quick_sort.cpp
    ../modules/read_infile.cpp
    ../modules/HMC.cpp
    ../modules/integrator.cpp
)
add_library(chemHMC STATIC ${chemHMC_header} ${chemHMC_scr})
target_link_libraries(chemHMC PUBLIC ${YAML_CPP_LIBRARIES} Kokkos::kokkos)
set_target_properties(chemHMC PROPERTIES LINKER_LANGUAGE CXX)
