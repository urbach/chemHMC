#pragma once

#include"coordinate.hh"
#include<random>

constexpr double pi() { return std::atan(1)*4; }

template<class URNG> void random_element(coordinate &c, URNG &engine) {

  std::uniform_real_distribution<double> dist(0, 1);

  c = coordinate(dist(engine), dist(engine), dist(engine));
  return;
}
