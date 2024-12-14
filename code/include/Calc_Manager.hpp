#ifndef CALC_MANAGER_HPP
#define CALC_MANAGER_HPP
#include "Calc.hpp"
#include <vector>
#include <memory>

class Calc_Manager {
private:
    std::vector<std::shared_ptr<Calc>> calc_list;

public:
    // Add a Calc object to the list
    void addCalc(std::shared_ptr<Calc> calc);

    // Initialize all Calc objects
    void initialize();
    void compute_force();
    void compute_potential();
};
#endif