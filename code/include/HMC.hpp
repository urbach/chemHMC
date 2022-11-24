#ifndef HMC_H
#define HMC_H
#include "read_infile.hpp"
#include "particles.hpp"

class HMC_class {
public:
    params_class *params;
    particles_type *particles;

    HMC_class(){};

    void init(int argc,char** argv);
       
    void run();

};
#endif // !HMC_H
