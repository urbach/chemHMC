#ifndef CALC_MANAGER_HPP
#define CALC_MANAGER_HPP
#include "Calc.hpp"
#include <vector>
#include <memory>
#include "particles.hpp"

class Calc_Manager {
private:
    std::vector<std::shared_ptr<Calc>> calc_list;
    

public:
    std::shared_ptr<particles_instance> particles;
    // Add a Calc object to the list
    void addCalc(std::shared_ptr<Calc> calc);

    // Initialize all Calc objects
    void initialize();
    void compute_force();
    double compute_potential();
    void set_particles(std::shared_ptr<particles_instance> particles_in);
    void print_timings();
};
#endif