#pragma once


//#include"random_element.hh"
#include<random>
#include<vector>
#include<cmath>
#include<fstream>
#include<complex>
#include<iostream>
#include<cassert>

using std::vector;

template<class T> class config {
public:
  using value_type = T;

  config(const size_t N, const size_t d) :
    N(N), d(d) {
    data.resize(N);
  }
  config(const config &U) :
    N(U.getN()), d(U.getd()) {
    data.resize(N);
#pragma omp parallel for
    for(size_t i = 0; i < getSize(); i++) {
      data[i] = U[i];
    }
  }

  size_t storage_size() const { return data.size() * sizeof(value_type); };
  size_t getN() const {
    return(N);
  }
  size_t getd() const {
    return(d);
  }
  size_t getSize() const {
    return(N);
  }

  void operator=(const config &U) {
    N = U.getN();
    data.resize(N);
#pragma omp parallel for
    for(size_t i = 0; i < getSize(); i++) {
      data[i] = U[i];
    }
  }

  //  value_type &operator()(size_t const index) {
  //    return data[ index ];
  //  }

  //  const value_type &operator()(size_t const index) {
  //    return data[ index ];
  //  }

  value_type &operator[](size_t const index) {
    return data[ index ];
  }

  const value_type &operator[](size_t const index) const {
    return data[index];
  }

  void save(std::string const &path) const;
  int load(std::string const &path);

private:
  size_t N, d;

  vector<value_type> data;

};

template<class T> void config<T>::save(std::string const &path) const {
  std::ofstream ofs(path, std::ios::out | std::ios::binary);
  ofs.write(reinterpret_cast<char const *>(data.data()), storage_size());
  return;
}

template<class T> int config<T>::load(std::string const &path) {
  std::cout << "## Reading config from file " << path << std::endl;
  std::ifstream ifs(path, std::ios::in | std::ios::binary);
  if(ifs) {
    ifs.read(reinterpret_cast<char *>(data.data()), storage_size());
    return 0;
  }
  else
    std::cerr << "Error: could not read file from " << path << std::endl;
  return 1;
}



template<class T> void  hotstart(config<T> & config, const int seed) {

  std::mt19937 engine(seed);

  for(size_t i = 0; i < config.getSize(); i++) {
    random_element(config[i], engine);
  }
}

