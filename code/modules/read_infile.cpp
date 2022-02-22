#define READ_INFILE_C

#include <array>
#include <cstring> 
#include <cstdlib>
#include <iterator>
#include <string>
#include <vector>
#include <sstream>
#include <fstream>
#include <iostream>

#include "read_infile.hpp"


template <typename Out>
void split(const std::string &s, char delim, Out result) {
    std::istringstream iss(s);
    std::string item;
    while (std::getline(iss, item, delim)) {
        *result++ = item;
    }
}

std::vector<std::string> split(const std::string &s, char delim) {
    std::vector<std::string> elems;
    split(s, delim, std::back_inserter(elems));
    std::vector<std::string>  r;
    std::string empty=" ";
    for (std::string s1: elems)
        if (s1.empty()==0){
            r.emplace_back(s1);
        }
            
    return r;
}
void read_par_int(std::fstream &newfile, std::string  name, int &p ){
        int match=0; 
        if (newfile.is_open()){ //checking whether the file is open
            std::string tp;
            while(getline(newfile, tp)){ //read data from file object and put it into string.
                std::vector<std::string> x = split(tp,'=');
                if(x.empty()==0){// if not empty line
                    if (x.size()!=2) {printf("error infile scan for %s\n expected:  param = value  \n found: %s \n",name.c_str(),tp.c_str()); exit(-9);  }
                        
                    std::vector<std::string> x0 = split(x[0],' ');
                    if (x0.size()!=1) {printf("error infile scan for %s\n param must be 1 word only \n found: %s ",name.c_str(), x[0].c_str()); exit(-9);  }
                    if (x0[0].compare(name)==0  ){
                        std::vector<std::string> rl = split(x[1],' ');
                        p=stoi(rl[0]);
                        match++;
                    }        
                }
            }
        }else{  printf("infile is not open \n");exit(-10);}
        
        if(match==0){  printf("could not find line %s = \n",name.c_str());exit(-10);} 
        if(match>1){  printf("multiple line %s = \n",name.c_str());exit(-10);} 
        // std::cout << name << " = "<< p  << endl;
        //rewind
        newfile.clear();
        newfile.seekg(0);
    }
void read_par_double(std::fstream &newfile, std::string  name, double &p ){
    int match=0; 
    if (newfile.is_open()){ //checking whether the file is open
        std::string tp;
        while(getline(newfile, tp)){ //read data from file object and put it into string.
            std::vector<std::string> x = split(tp,'=');
            if(x.empty()==0){// if not empty line
                if (x.size()!=2) {printf("error infile scan for %s\n expected:  param = value  \n found: %s \n",name.c_str(),tp.c_str()); exit(-9);  }
                    
                std::vector<std::string> x0 = split(x[0],' ');
                if (x0.size()!=1) {printf("error infile scan for %s\n param must be 1 word only \n found: %s ",name.c_str(), x[0].c_str()); exit(-9);  }
                if (x0[0].compare(name)==0  ){
                    std::vector<std::string> rl = split(x[1],' ');
                    p=stod(rl[0]);
                    match++;
                }        
            }
        }
    }else{  printf("infile is not open \n");exit(-10);}
    
    if(match==0){  printf("could not find line %s = \n",name.c_str());exit(-10);} 
    if(match>1){  printf("multiple line %s = \n",name.c_str());exit(-10);} 
    // std::cout << name << " = "<< p  << endl;

    //rewind
    newfile.clear();
    newfile.seekg(0);
}  
void read_par_string(std::fstream &newfile, std::string  name, std::string &s , bool required=true ){
    int match=0; 
    if (newfile.is_open()){ //checking whether the file is open
        std::string tp;
        while(getline(newfile, tp)){ //read data from file object and put it into string.
            std::vector<std::string> x = split(tp,'=');
            if(x.empty()==0){// if not empty line
                if (x.size()!=2) {printf("error infile scan for %s\n expected:  param = value  \n found: %s \n",name.c_str(),tp.c_str()); exit(-9);  }
                    
                std::vector<std::string> x0 = split(x[0],' ');
                if (x0.size()!=1) {printf("error infile scan for %s\n param must be 1 word only \n found: %s ",name.c_str(), x[0].c_str()); exit(-9);  }
                if (x0[0].compare(name)==0  ){
                    std::vector<std::string> rl = split(x[1],' ');
                    s= rl[0];
                    match++;
                }        
            }
        }
    }else{  printf("infile is not open \n");exit(-10);}
    
    if(match==0){  
        if(required){ std::cout << "could not find line: "<< name.c_str() << " =" << std::endl; exit(-10);}
        else 
            std::cout << "could not find param: " << name << "\n default falue: "<< name << " = "<<  s << "" << std::endl;
    } 
    if(match>1 ){  printf("multiple line %s = \n",name.c_str());exit(-10);} 
    // std::cout << name << " = "<< s  << endl;
    //rewind
    newfile.clear();
    newfile.seekg(0);
}

DataContainer::DataContainer(int argc, char** argv ){
    int opt = -1;
    char infilename[200];

    // search for command line option and put filename in "infilename"
    for(int i = 0; i < argc; ++i) {
      if(std::strcmp(argv[i], "-i") == 0) {
        opt = i+1;
        break;
      }
    }
    if(opt < 0) {
      std::cout << "No input file specified, Aborting" << std::endl;
      exit(1);
    } else {
      sprintf(infilename, "%s", argv[opt]);
      std::cout << "Trying input file " << infilename << std::endl;
    }
    std::fstream newfile;
   
    newfile.open(infilename,std::ios::in); 
    
    // open file for reading
    if (!newfile.is_open()) {
      std::cerr << "Could not open file " << infilename << std::endl;
      std::cerr << "Aborting..." << std::endl;
      exit(-10);
    }
    read_par_double(newfile,"L",L[0]);
    L[1]=L[0];
    L[2]=L[0];
    read_par_int(newfile,"Nathoms",Nathoms);

}