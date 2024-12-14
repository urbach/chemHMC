#include "LJ.hpp"
#include <iostream>
#include <cmath> // For potential calculations

void LJ::init() {
    epsilon = 1.5;
    sigma = 1.0;
    std::cout << "Lennard-Jones initialized with epsilon = " << epsilon
              << " and sigma = " << sigma << std::endl;
}

double LJ::potential() {
    std::cout << "Calculating Lennard-Jones potential..." << std::endl;
    double r = 2.0;
    double sr = sigma / r;
    double sr6 = std::pow(sr, 6);
    return 4.0 * epsilon * (sr6 * sr6 - sr6);
}

double LJ::force() {
    std::cout << "Calculating Lennard-Jones force..." << std::endl;
    double r = 2.0;
    double sr = sigma / r;
    double sr6 = std::pow(sr, 6);
    return -24.0 * epsilon * (2.0 * sr6 * sr6 - sr6) / r;
}