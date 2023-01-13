#include"config.hh"
#include"potential_energy.hh"
#include"coordinate.hh"
#include"parse_commandline.hh"
#include"version.hh"

#include<iostream>
#include<fstream>
#include<iomanip>
#include<sstream>
#include<random>
#include<boost/program_options.hpp>

namespace po = boost::program_options;
using std::cout;
using std::endl;

int main(int ac, char* av[]) {
  general_params gparams;
  size_t N_rev;
  size_t n_steps;
  size_t exponent;
  double tau;
  size_t integs;

  cout << "## chemHMC" << endl;
  cout << "## (C) Carsten Urbach <urbach@hiskp.uni-bonn.de> (2023)" << endl;
  cout << "## GIT branch " << GIT_BRANCH << " on commit " << GIT_COMMIT_HASH << endl << endl;  

  //  po::options_description desc("Allowed options");
  //  add_general_options(desc, gparams);
  // add HMC specific options
  //  desc.add_options()
  //    ("nrev", po::value<size_t>(&N_rev)->default_value(0), "frequenz of reversibility tests N_rev, 0: not reversibility test")
  //    ("nsteps", po::value<size_t>(&n_steps)->default_value(1000), "n_steps")
  //    ("tau", po::value<double>(&tau)->default_value(1.), "trajectory length tau")
  //    ("exponent", po::value<size_t>(&exponent)->default_value(0), "exponent for rounding")
  //    ("integrator", po::value<size_t>(&integs)->default_value(0), "itegration scheme to be used: 0=leapfrog, 1=lp_leapfrog, 2=omf4, 3=lp_omf4, 4=Euler, 5=RUTH, 6=omf2")
  //    ;

  //  int err = parse_commandline(ac, av, desc, gparams);
  //  if(err > 0) {
  //    return err;
  //  }

  config<coordinate> U(10, 3);
  if(gparams.restart*0) {
    int err = U.load(gparams.configfilename);
    if(err != 0) {
      return err;
    }
  }
  else{
    hotstart(U, gparams.seed);
  }

  double E = potential_energy(U);

  std::cout << "Energy is " << E << std::endl;
  
  return(0);
}
