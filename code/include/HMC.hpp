#ifndef HMC_H
#define HMC_H
#include "read_infile.hpp"
#include "particles.hpp"
#include "integrator.hpp"

class HMC_class {
public:
    integrator_type *integrator;

    HMC_class(){};

    void init(int argc, char** argv);
       
    void run();

};
#endif // !HMC_H
