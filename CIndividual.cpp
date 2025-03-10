#include "CIndividual.h"
CIndividual::CIndividual() {
	this->fitness = 0.0;
	this->matrix = Matrix();
}

CIndividual::CIndividual(const CIndividual& cOther) {
	this->fitness = cOther.fitness;
	this->matrix = Matrix(cOther.matrix);
	this->genotype = cOther.genotype;
}

CIndividual::CIndividual(std::vector<int>& vec, Matrix& matrix) {
	this->fitness = 0.0;
	this->matrix = Matrix(matrix);
	for (int i = 0; i < vec.size(); ++i) {
		this->genotype.push_back(vec[i]);
	}
}

CIndividual& CIndividual::operator=(const CIndividual& cOther) {
	if (this != &cOther) {
		this->fitness = cOther.fitness;
		this->matrix = Matrix(cOther.matrix);
		this->genotype = cOther.genotype;
	}
	return *this;
}

double CIndividual::getFitness() {
	return this->fitness;
}

std::vector<int>& CIndividual::getGenotype() {
	return this->genotype;
}

Matrix& CIndividual::getMatrix() {

	return this->matrix;
}

void CIndividual::setFitness(double f) {
	this->fitness = f;
}
