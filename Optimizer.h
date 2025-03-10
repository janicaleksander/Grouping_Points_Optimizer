#ifndef OPTIMIZER_H
#define OPTIMIZER_H

#include "GroupingEvaluator.h"

#include <iostream>
#include <numeric>
#include <random>
#include <vector>
#include <thread>
#include <mutex>
#include <vector>
#include <iostream>
#include <random>
#include <memory>
#include <atomic>
#include "CMySmartPointer.h"
#include "Matrix.h"
#include "CIndividual.h"
#include <vector>
#include <utility> // for std::pair
#include "Matrix.h" 
#include "CMySmartPointer.h" 

struct Change {
	int pointIdx;
	int oldCluster;
	int newCluster;
};

using namespace std;

namespace NGroupingChallenge
{
	class COptimizer
	{
	public:
		COptimizer(CGroupingEvaluator& cEvaluator);

		void vInitialize();
		void vRunIteration();

		vector<int> pvGetCurrentBest() { return v_current_best.getGenotype(); }
		CMySmartPointer<CIndividual> methodIVCrossoverAlgorithm(CIndividual& fst, CIndividual& snd);
		CMySmartPointer<CIndividual> methodVCrossoverAlgorithm(CIndividual& fst, CIndividual& snd);
		void runIsland(std::vector<CIndividual>& pop, int& better, bool& isBetter);

		std::vector<CMySmartPointer<CIndividual>> susSelectionAlgorithm(std::vector<CIndividual>& population, int toNextGeneration);
		CMySmartPointer<CIndividual> methodPXCrossoverAlgorithm(CIndividual& fst, CIndividual& snd);
		CMySmartPointer<CIndividual> methodIMutationAlgorithm(CIndividual& fst);
		CMySmartPointer<CIndividual> methodIIMutationAlgorithm(CIndividual& fst);
		CMySmartPointer<CIndividual> methodILocalAlgorithm(CIndividual& fst);

		CMySmartPointer<Matrix> createVIG(const std::vector<CPoint>& points, std::vector<int>& genotype);

		CMySmartPointer<Matrix> createPointsDistanceMatrix(const std::vector<CPoint>& points);
		std::pair<double, CMySmartPointer<Matrix>> updateVIGMatrix(
			const Matrix& oldVIG,
			std::vector<int>& c,
			const std::vector<Change>& changes,
			double oldFitness);
		//std::pair<double, CMySmartPointer<Matrix>> updateVIGMatrix(Matrix& oldVIG, std::vector<int>& c, std::vector<int> changedIdx, double oldFitness);

	private:
		CGroupingEvaluator& c_evaluator;
		mt19937 c_random_engine;

		std::mutex best_solution_mutex;
		double d_current_best_fitness;
		CIndividual v_current_best;
		CMySmartPointer<Matrix> pointsDistanceMatrix;
		double distPointsDistanceMatrix;

		double computeFitnessChange(int idx, int oldCluster, int newCluster, const std::vector<int>& c, const Matrix& vig);
		//added;
		std::vector<CIndividual> population;
		std::vector<CIndividual> island1;
		std::vector<CIndividual> island2;
		std::vector<CIndividual> island3;


		int populationSize;
		int numOfIslands;
		int numberOfClusters;
		int whenMigration;
		int globalIteration;
		int toMigrate;
		double propabilityMinCross;
		double propabilityMaxCross;
		double propabilityMinMut;
		double propabilityMaxMut;
		int maxGeneration;
		int maxIteration;
		int currentGeneration;


		bool isBetter1;
		bool isBetter2;
		bool isBetter3;
		int better1;
		int better2;
		int better3;


		double betterdEvaluate(const int* piSolution, const std::vector<CPoint>& v_points) const;
		double betterdEvaluate(const vector<int>* pvSolution, const std::vector<CPoint>& v_points) const;
		double betterdEvaluate(const vector<int>& vSolution, const std::vector<CPoint>& v_points) const;
	};
}

#endif