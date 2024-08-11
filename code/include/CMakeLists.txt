
set(chemHMC_header
    HMC.hpp
    read_infile.hpp
    integrator.hpp
    particles_type.hpp
    particles.hpp
    # binning.hpp
    # identical_particles.hpp
    # quick_sort.hpp
)

set(chemHMC_scr 
    ../modules/HMC.cpp
    ../modules/read_infile.cpp
    ../modules/integrator.cpp
    ../modules/particles_type.cpp
    ../modules/particles.cpp
    #../modules/identical_particles_binning.cpp
    # ../modules/binning.cpp
    #../modules/identical_particles.cpp
    # ../modules/quick_sort.cpp
)
add_library(chemHMC STATIC ${chemHMC_header} ${chemHMC_scr})
target_link_libraries(chemHMC PUBLIC ${YAML_CPP_LIBRARIES} Kokkos::kokkos)
set_target_properties(chemHMC PROPERTIES LINKER_LANGUAGE CXX)
