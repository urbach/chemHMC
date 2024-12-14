#include "Calc_Manager.hpp"
#include <iostream>

void Calc_Manager::addCalc(std::shared_ptr<Calc> calc) {
    calc_list.push_back(calc);
}

void Calc_Manager::initialize() {
    for (const auto& calc : calc_list) {
        if (calc) { // Check if the pointer is valid
            calc->init();
        }
    }
}

void Calc_Manager::compute_force() {
    for (const auto& calc : calc_list) {
            calc->force();
    }
}

void Calc_Manager::compute_potential() {
    for (const auto& calc : calc_list) {
            calc->potential();
    }
}