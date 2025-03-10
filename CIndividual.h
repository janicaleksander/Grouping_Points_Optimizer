#pragma once
#include <vector>
#include "Matrix.h"

class CIndividual {
public:
    CIndividual(); 
    CIndividual(const CIndividual& cOther);  
    CIndividual& operator=(const CIndividual& other); 
    CIndividual(std::vector<int>& vec, Matrix& matrix); 

    double getFitness();  
    std::vector<int>& getGenotype();  
    Matrix& getMatrix();  
    void setFitness(double f);  

private:
    Matrix matrix; 
    std::vector<int> genotype;
    double fitness;
};
