
set(chemHMC_header
    HMC.hpp
    ../modules/integrators/integrator_type.hpp
    ../modules/integrators/leapfrog.hpp
    ../modules/integrators/OMF2.hpp
    ../modules/integrators/OMF4.hpp
    ../modules/integrators/velocity_verlet.hpp
    ../modules/integrators/integrator.hpp
    ../modules/integrators/SHAKE.hpp
    Input_reader.hpp
    Parameters.hpp
    particles.hpp
    atom.hpp
    bonds.hpp
    ../modules/potentials/non_bonded_interactions/coulomb_ewald.hpp
    ../modules/potentials/non_bonded_interactions/coulomb_pppm.hpp
    ../modules/potentials/non_bonded_interactions/coulomb_direct.hpp
    ../modules/potentials/non_bonded_interactions/LJ.hpp
    Calc.hpp
    Calc_Manager.hpp
    ../modules/neighbor_list/Neighbor_list.hpp
)

set(chemHMC_scr 
    ../modules/HMC.cpp
    ../modules/potentials/non_bonded_interactions/LJ.cpp
    ../modules/potentials/non_bonded_interactions/coulomb_ewald.cpp
    ../modules/potentials/non_bonded_interactions/coulomb_pppm.cpp
    ../modules/potentials/non_bonded_interactions/coulomb_direct.cpp
    ../modules/Calc_Manager.cpp
    ../modules/Input_reader.cpp
    ../modules/integrators/integrator_type.cpp
    ../modules/integrators/leapfrog.cpp
    ../modules/integrators/OMF2.cpp
    ../modules/integrators/OMF4.cpp
    ../modules/integrators/velocity_verlet.cpp
    ../modules/integrators/SHAKE.cpp
    ../modules/particles/particles.cpp
    ../modules/neighbor_list/Neighbor_list.cpp
    ../modules/potentials/molecules/bonds.cpp
    ../modules/neighbor_list/Neighbor_list_bonds.cpp
    ../modules/neighbor_list/Neighbor_list_cell.cpp
)
add_library(chemHMC STATIC ${chemHMC_header} ${chemHMC_scr})
target_link_libraries(chemHMC PUBLIC ${YAML_CPP_LIBRARIES} Kokkos::kokkos)
set_target_properties(chemHMC PROPERTIES LINKER_LANGUAGE CXX)
