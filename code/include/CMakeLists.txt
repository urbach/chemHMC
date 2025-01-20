
set(chemHMC_header
    HMC.hpp
    integrator.hpp
    Input_reader.hpp
    Parameters.hpp
    particles.hpp
    atom.hpp
    bonds.hpp
    coulomb.hpp
    ../modules/potentials/non_bonded_interactions/LJ.hpp
    Calc.hpp
    Calc_Manager.hpp
    Neighbor_list.hpp
)

set(chemHMC_scr 
    ../modules/HMC.cpp
    ../modules/potentials/non_bonded_interactions/LJ.cpp
    ../modules/Calc_Manager.cpp
    ../modules/Input_reader.cpp
    ../modules/integrator.cpp
    ../modules/particles/particles.cpp
    ../modules/Neighbor_list.cpp
    ../modules/potentials/molecules/bonds.cpp
    ../modules/Neighbor_list.cpp
    ../modules/Neighbor_list_bonds.cpp
    ../modules/particles/non_bonded_interactions/coulomb.cpp
    ../modules/potentials/non_bonded_interactions/coulomb.cpp
)
add_library(chemHMC STATIC ${chemHMC_header} ${chemHMC_scr})
target_link_libraries(chemHMC PUBLIC ${YAML_CPP_LIBRARIES} Kokkos::kokkos)
set_target_properties(chemHMC PROPERTIES LINKER_LANGUAGE CXX)
