
set(chemHMC_header
    HMC.hpp
    read_infile.hpp
    integrator.hpp
    particles_type.hpp
    particles.hpp
    atom.hpp
    LJ.hpp
    Calc.hpp
    Calc_Manager.hpp
    # binning.hpp
    # identical_particles.hpp
    # quick_sort.hpp
)

set(chemHMC_scr 
    ../modules/HMC.cpp
    ../modules/LJ.cpp
    ../modules/Calc_Manager.cpp
    ../modules/read_infile.cpp
    ../modules/integrator.cpp
    ../modules/particles/particles_type.cpp
    ../modules/particles/particles.cpp
    ../modules/particles/non_bonded_interactions/lennard_jones.cpp
    ../modules/particles/non_bonded_interactions/coulomb.cpp
    ../modules/particles/bonds/bonds.cpp
    ../modules/particles/bonds/opls.cpp
    #../modules/identical_particles_binning.cpp
    # ../modules/binning.cpp
    #../modules/identical_particles.cpp
    # ../modules/quick_sort.cpp
)
add_library(chemHMC STATIC ${chemHMC_header} ${chemHMC_scr})
target_link_libraries(chemHMC PUBLIC ${YAML_CPP_LIBRARIES} Kokkos::kokkos)
set_target_properties(chemHMC PROPERTIES LINKER_LANGUAGE CXX)
