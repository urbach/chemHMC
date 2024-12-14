#ifndef CALC_HPP
#define CALC_HPP
class Calc {
public:
    virtual void init() = 0;
    virtual double potential() = 0;
    virtual double force() = 0;
    virtual ~Calc() {}
};
#endif