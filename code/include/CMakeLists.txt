
set(chemHMC_header
    particles.hpp
    binning.hpp
    identical_particles.hpp
    read_infile.hpp
    HMC.hpp
    integrator.hpp
    # quick_sort.hpp
)

set(chemHMC_scr
    ../modules/particles.cpp
    ../modules/binning.cpp
    ../modules/identical_particles.cpp
    ../modules/identical_particles_binning.cpp
    ../modules/quick_sort.cpp
    ../modules/read_infile.cpp
    ../modules/HMC.cpp
    ../modules/integrator.cpp
)
add_library(chemHMC STATIC ${chemHMC_header} ${chemHMC_scr})
target_link_libraries(chemHMC PUBLIC ${YAML_CPP_LIBRARIES} fmt::fmt Kokkos::kokkos)
set_target_properties(chemHMC PROPERTIES LINKER_LANGUAGE CXX)
