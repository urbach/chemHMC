#pragma once

class coordinate {
public:
  coordinate() {
    r[0] = 0.;
    r[1] = 0.;
    r[2] = 0.;
  }
  coordinate(const double r0, const double r1, const double r2) {
    r[0] = r0;
    r[1] = r1;
    r[2] = r2;
  }
  coordinate(const coordinate& _c) {
    for(size_t i = 0; i < 3; i++) {
      r[i] = _c[i];
    }
  }
  coordinate(const double* _r) {
    for(size_t i = 0; i < 3; i++) {
      r[i] = _r[i];
    }
  }

  void operator=(const coordinate &c) {
    for(size_t i = 0; i < 3; i++) {
      r[i] = c[i];
    }
  }
  
  double &operator[](size_t const index) {
    return r[ index ];
  }

  const double &operator[](size_t const index) const {
    return r[index];
  }

  double norm() {
    return(sqrt(r[0]*r[0] + r[1]*r[1] + r[2]*r[2]));
  }
  
private:
  double r[3];
};

inline coordinate operator+(const coordinate& c1, const coordinate& c2) {
  return(coordinate(c1[0]+c2[0], c1[1]+c2[1], c1[2]+c2[2]));
}

inline coordinate operator-(const coordinate& c1, const coordinate& c2) {
  return(coordinate(c1[0]-c2[0], c1[1]-c2[1], c1[2]-c2[2]));
}

