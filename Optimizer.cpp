#include "Optimizer.h"
#include "map"
#include <algorithm>
#include <cmath>

#include <thread>
#include <mutex>
#include <random>
#include <vector>
#include <iostream>#include <iostream>                 // std::cout, std::endl
#include <vector>                   // std::vector
#include <random>                   // std::random_device, std::default_random_engine, std::uniform_real_distribution, std::uniform_int_distribution
#include <future>                   // std::async, std::future
#include <algorithm>                // std::for_each (opcjonalnie, jeœli u¿yjemy w innych miejscach)
#include <numeric>                  // std::iota (opcjonalnie, jeœli w przysz³oœci potrzebujesz)
#include <unordered_map>
#include <optional>
#include <unordered_map>
#include <sstream>
#include "Matrix.h"
#include "CIndividual.h"
#include "CMySmartPointer.h"
#include <unordered_set>
using namespace NGroupingChallenge;

COptimizer::COptimizer(CGroupingEvaluator& cEvaluator)
	: c_evaluator(cEvaluator)
{
	random_device c_seed_generator;
	c_random_engine.seed(c_seed_generator());
}

void COptimizer::vInitialize()
{

	numeric_limits<double> c_double_limits;
	d_current_best_fitness = c_double_limits.max();

	v_current_best.getGenotype().clear();

	//this->populationSize = 300;
	//this->populationSize = 180;
	//this->populationSize = 60;
	this->populationSize = 57;
	//this->maxGeneration = 15;
	this->maxGeneration = 10;
	//this->whenMigration = 4;
	this->whenMigration = 8;
	this->maxIteration = this->populationSize / 6;
	this->numOfIslands = 3;
	this->propabilityMinCross = .5;
	this->propabilityMaxCross = .9;
	this->propabilityMinMut = .5;
	this->propabilityMaxMut = .95;
	//this->toMigrate = 10;
	//this->toMigrate = 7;
	this->toMigrate = 7;
	this->globalIteration = 0;
	this->numberOfClusters = this->c_evaluator.iGetUpperBound();



	//auto pointsVec = this->c_evaluator.vGetPoints();
	//this->points = pointsVec;

	auto v = createPointsDistanceMatrix(this->c_evaluator.vGetPoints());
	this->pointsDistanceMatrix = v;

	uniform_int_distribution<int> c_candidate_distribution(c_evaluator.iGetLowerBound(), c_evaluator.iGetUpperBound());
	for (int i = 0; i < 2 * populationSize; ++i) {
		std::vector<int> vi;
		for (int j = 0; j < c_evaluator.iGetNumberOfPoints(); ++j) {
			vi.push_back(c_candidate_distribution(c_random_engine));
		}
		auto vig = (createVIG(this->c_evaluator.vGetPoints(), vi));
		this->population.push_back(CIndividual(vi, *vig));
	}

	for (auto& v : this->population) {
		v.setFitness(betterdEvaluate(v.getGenotype(), this->c_evaluator.vGetPoints()));
	}
	auto vec = (susSelectionAlgorithm(this->population, this->populationSize));

	this->population.clear();
	this->population.shrink_to_fit();
	for (auto& v : vec) {
		this->population.push_back((*v));
	}




	int toChoose = this->populationSize / 3;
	int  i = 0;
	int k = 0;
	for (i; i < toChoose; ++i) {
		this->island1.push_back(this->population[i]);
	}
	k = i;
	i = 0;
	for (i; i < toChoose; ++i) {
		this->island2.push_back(this->population[i]);
	}
	k = i;
	i = 0;
	for (i; i < toChoose; ++i) {
		this->island3.push_back(this->population[i]);
	}

	isBetter1 = false;
	isBetter2 = false;
	isBetter3 = false;
	better1 = 0;
	better2 = 0;
	better3 = 0;


}


double P_dynamic(int t, int T_max, double P_max, double P_min) {
	double alpha = -log(P_min / P_max) / (0.75 * T_max);
	double beta = -log(P_min / P_max) / (0.25 * T_max);

	if (t < 3 * T_max / 4) {
		return P_max * exp(-alpha * t);
	}
	else {
		return P_min + (P_max - P_min) * exp(-beta * (t - 3 * T_max / 4));
	}
}


void COptimizer::vRunIteration() {



	bool anyImprovement1 = false;
	bool anyImprovement2 = false;
	bool anyImprovement3 = false;



	std::thread t1([&]() {
		runIsland(this->island1, this->better1, this->isBetter1);
		if (this->isBetter1) anyImprovement1 = true;
		});

	std::thread t2([&]() {
		runIsland(this->island2, this->better2, isBetter2);
		if (this->isBetter2) anyImprovement2 = true;

		});

	std::thread t3([&]() {
		runIsland(this->island3, this->better3, isBetter3);
		if (this->isBetter3) anyImprovement3 = true;

		});



	// Wait for all threads to complete
	t1.join();
	t2.join();
	t3.join();

	if (!anyImprovement1) {
		this->better1++;
	}
	if (!anyImprovement2) {
		this->better2++;
	}
	if (!anyImprovement3) {
		this->better3++;
	}
	// After all threads complete, handle migration
	{
		std::lock_guard<std::mutex> lock(best_solution_mutex);
		this->globalIteration++;

		if (this->globalIteration == this->whenMigration) {
			//1->2
			for (int i = 0; i < this->toMigrate; ++i) {
				island2[i] = island1[i];
			}
			//2->3
			for (int i = 0; i < this->toMigrate; ++i) {
				island3[i] = island2[i];
			}
			//3->1
			for (int i = 0; i < this->toMigrate; ++i) {
				island1[i] = island3[i];
			}
			this->globalIteration = 0;
		}
	}
	// After all threads complete, handle migration

	std::cout << this->better1 << "\n";
	std::cout << this->better2 << "\n";
	std::cout << this->better3 << "\n";

	//std::cout << this->d_current_best_fitness << "\n";
	std::cout << this->c_evaluator.dEvaluate(this->v_current_best.getGenotype());
}

void COptimizer::runIsland(std::vector<CIndividual>& islandPopulation, int& better, bool& isBetter) {
	CIndividual bestIndividual(islandPopulation[0]);
	std::random_device rd;
	std::default_random_engine generator(rd());
	std::uniform_real_distribution<double> distribution(0.0, 1.0);
	std::uniform_int_distribution<int> distribution2(0, (islandPopulation.size() - 1));
	std::mutex end_mutex; // Mutex for thread-safe access to `end`

	int currentIteration = 0;
	int currentGeneration = 0;

	while (currentGeneration < this->maxGeneration) {
		bool isBestChange = false;
		currentGeneration++;
		int T = 1;
		std::vector<CMySmartPointer<CIndividual>> end;

		// Define thread function
		auto thread_task = [&](int start, int end_limit) {
			std::uniform_int_distribution<int> distribution2(0, (islandPopulation.size() - 1));
			while (true) {
				// Lock mutex to safely update shared resources
				int local_T;
				{
					std::lock_guard<std::mutex> lock(end_mutex);
					if (T >= end_limit) break;
					local_T = T++;
				}

				double p_dynamic = P_dynamic(currentGeneration, this->maxGeneration, this->propabilityMaxCross, this->propabilityMinCross);
				double rnd1 = distribution(generator);
				double rnd2 = distribution(generator);
				double rnd3 = distribution(generator);
				double rnd4 = distribution(generator);
				CMySmartPointer<CIndividual> result;

				if (rnd1 > p_dynamic) {
					CMySmartPointer<CIndividual> afterCrossoverChild = methodPXCrossoverAlgorithm(islandPopulation[distribution2(generator)], islandPopulation[distribution2(generator)]);
					CMySmartPointer<CIndividual> afterFirstMutation = methodIMutationAlgorithm(*afterCrossoverChild);
					CMySmartPointer<CIndividual> afterSecondMutation = methodIIMutationAlgorithm(*afterFirstMutation);
					result = (methodILocalAlgorithm(*afterSecondMutation));
				}
				else if (rnd2 > p_dynamic) {
					CMySmartPointer<CIndividual> afterFirstMutation = methodIMutationAlgorithm(islandPopulation[distribution2(generator)]);
					CMySmartPointer<CIndividual> afterSecondMutation = methodIIMutationAlgorithm(*afterFirstMutation);
					result = (methodILocalAlgorithm(*afterSecondMutation));

				}
				else if (rnd3 > p_dynamic) {
					CMySmartPointer<CIndividual> afterSecondMutation = methodIIMutationAlgorithm(islandPopulation[distribution2(generator)]);
					result = (methodILocalAlgorithm(*afterSecondMutation));

				}
				else {
					CMySmartPointer<CIndividual> afterCrossoverChild = methodVCrossoverAlgorithm(islandPopulation[distribution2(generator)], islandPopulation[distribution2(generator)]);
					result = (methodILocalAlgorithm(*afterCrossoverChild));

				}
				// Add the result to `end`
				{

					std::lock_guard<std::mutex> lock(end_mutex);
					//	end.push_back((methodILocalAlgorithm(*result)));
					end.push_back(((result)));

				}
			}

			};



		// Launch threads
		const int num_threads = 5;
		std::vector<std::thread> threads;
		for (int i = 0; i < num_threads; ++i) {
			threads.emplace_back(thread_task, 1, islandPopulation.size());
		}

		// Join threads
		for (auto& thread : threads) {
			if (thread.joinable()) {
				thread.join();
			}
		}

		//std::cout << "rozmiar end:" << end.size() << "\n";
		for (auto& v_candidate : end) {
			//double d_candidate_fitness = c_evaluator.dEvaluate(v_candidate);
			double d_candidate_fitness = (*v_candidate).getFitness();
			{
				std::lock_guard<std::mutex> lock(best_solution_mutex);
				if (d_candidate_fitness < d_current_best_fitness) {
					v_current_best = *(v_candidate);
					d_current_best_fitness = d_candidate_fitness;
					bestIndividual = v_current_best;
					better = 0;
					isBetter = true;
					isBestChange = true;
				}

			}
		}

		if (!isBestChange) {
			std::lock_guard<std::mutex> lock(best_solution_mutex);
			bestIndividual = this->v_current_best;
		}


		islandPopulation.clear();
		islandPopulation.shrink_to_fit();
		if (better < 25) {
			for (int i = 0; i < end.size() / 2; ++i) {
				islandPopulation.push_back(bestIndividual);
			}
		}
		else {
			for (int i = 0; i < end.size() / 2; ++i) {
				islandPopulation.push_back(*methodILocalAlgorithm(bestIndividual));
			}

		}
		int i = 0;
		std::uniform_int_distribution<int> c_candidate_distribution(c_evaluator.iGetLowerBound(), c_evaluator.iGetUpperBound());
		while (islandPopulation.size() < this->populationSize / this->numOfIslands) {
			/*

			*/
			if (i < this->population.size() - 1) {
				islandPopulation.push_back(this->population[i]);
			}
			else {
				std::vector<int> v;
				for (int j = 0; j < c_evaluator.iGetNumberOfPoints(); ++j) {
					v.push_back(c_candidate_distribution(generator));
				}
				double f = betterdEvaluate(v, this->c_evaluator.vGetPoints());
				auto c = CIndividual(v, (*(createVIG(this->c_evaluator.vGetPoints(), v))));
				c.setFitness(f);
				islandPopulation.push_back(c);
			}
			i++;
		}


		//std::cout << "roxmiar this po:" << islandPopulation.size() << "\n";
	}

	if (!isBetter) {
		better++;
	}

}


CMySmartPointer<CIndividual> NGroupingChallenge::COptimizer::methodIVCrossoverAlgorithm(CIndividual& fst, CIndividual& snd) {
	std::vector<int> child(fst.getGenotype().size());

	std::srand(static_cast<unsigned int>(std::time(nullptr)));

	size_t point1 = std::rand() % fst.getGenotype().size();
	size_t point2 = std::rand() % fst.getGenotype().size();

	if (point1 > point2) {
		std::swap(point1, point2);
	}

	for (size_t i = 0; i < point1; ++i) {
		(child)[i] = fst.getGenotype()[i];
	}

	for (size_t i = point1; i <= point2; ++i) {
		(child)[i] = snd.getGenotype()[i];
	}

	for (size_t i = point2 + 1; i < fst.getGenotype().size(); ++i) {
		(child)[i] = fst.getGenotype()[i];
	}


	double f = betterdEvaluate(child, this->c_evaluator.vGetPoints());
	CIndividual* c = new CIndividual((child), (*(createVIG(this->c_evaluator.vGetPoints(), child))));
	c->setFitness(f);
	return CMySmartPointer<CIndividual>(c);
}


CMySmartPointer<CIndividual> NGroupingChallenge::COptimizer::methodVCrossoverAlgorithm(CIndividual& fst, CIndividual& snd) {
	// Extract genotypes from the parents
	std::vector<int> childA(fst.getGenotype().size());
	std::vector<int> childB(snd.getGenotype().size());

	// Seed random number generator
	std::srand(static_cast<unsigned int>(std::time(nullptr)));

	// Generate two random crossover points
	size_t point1 = std::rand() % fst.getGenotype().size();
	size_t point2 = std::rand() % fst.getGenotype().size();

	// Ensure point1 < point2
	if (point1 > point2) {
		std::swap(point1, point2);
	}

	// Copy first segment from fst to childA, snd to childB
	for (size_t i = 0; i < point1; ++i) {
		childA[i] = fst.getGenotype()[i];
		childB[i] = snd.getGenotype()[i];
	}

	// Copy middle segment from snd to childA, fst to childB
	for (size_t i = point1; i <= point2; ++i) {
		childA[i] = snd.getGenotype()[i];
		childB[i] = fst.getGenotype()[i];
	}

	// Copy last segment from fst to childA, snd to childB
	for (size_t i = point2 + 1; i < fst.getGenotype().size(); ++i) {
		childA[i] = fst.getGenotype()[i];
		childB[i] = snd.getGenotype()[i];
	}

	// Evaluate fitness for both children
	double fitnessA = betterdEvaluate(childA, this->c_evaluator.vGetPoints());
	double fitnessB = betterdEvaluate(childB, this->c_evaluator.vGetPoints());

	// Create individuals
	CIndividual* cA = new CIndividual(childA, *(createVIG(this->c_evaluator.vGetPoints(), childA)));
	CIndividual* cB = new CIndividual(childB, *(createVIG(this->c_evaluator.vGetPoints(), childB)));

	cA->setFitness(fitnessA);
	cB->setFitness(fitnessB);

	// Return the child with the lower fitness value
	if (fitnessA < fitnessB) {
		delete cB; // Free the unused individual
		return CMySmartPointer<CIndividual>(cA);
	}
	else {
		delete cA; // Free the unused individual
		return CMySmartPointer<CIndividual>(cB);
	}
}

CMySmartPointer<Matrix> COptimizer::createVIG(const std::vector<CPoint>& points, std::vector<int>& genotype) {
	if (points.empty()) {
		return CMySmartPointer<Matrix>(new Matrix());
	}

	int n = points.size();
	double averageDistance = (n > 1) ? ((this->distPointsDistanceMatrix) / (n * (n - 1))) : 0.0;
	CMySmartPointer<Matrix> VIGMatrix(new Matrix(n, n));
	for (int i = 0; i < n - 1; ++i) {
		for (int j = i + 1; j < n; ++j) {
			double pointDist = (this->pointsDistanceMatrix->getElem(i, j));
			if (genotype[i] == genotype[j] || pointDist < 0.1 * averageDistance) {
				VIGMatrix->setOn(i, j, 1);
				VIGMatrix->setOn(j, i, 1);
			}
		}
		VIGMatrix->setOn(i, i, 0);
	}


	return VIGMatrix;
}


CMySmartPointer<Matrix>  NGroupingChallenge::COptimizer::createPointsDistanceMatrix(const std::vector<CPoint>& points) {
	int n = points.size();
	double dist = 0.0;
	CMySmartPointer<Matrix> precomputedDistances(new Matrix(n, n));
	for (int i = 0; i < n - 1; ++i) {
		for (int j = i + 1; j < n; ++j) {
			double pointDist = points[i].dCalculateDistance(points[j]);
			precomputedDistances->setOn(i, j, pointDist);
			precomputedDistances->setOn(j, i, pointDist);
			dist += pointDist;
		}
	}
	this->distPointsDistanceMatrix = dist;
	return precomputedDistances;
}



CMySmartPointer<CIndividual> NGroupingChallenge::COptimizer::methodIMutationAlgorithm(CIndividual& fst) {
	CMySmartPointer<std::vector<int>> fstCOPY(new std::vector<int>(fst.getGenotype()));

	std::map<int, int> clusterCounts;
	for (auto v : *fstCOPY) {
		clusterCounts[v]++;
	}

	if (clusterCounts.size() < 2) {
		return CMySmartPointer<CIndividual>(new CIndividual(fst));
	}

	std::vector<std::pair<int, int>> clusterSizes(clusterCounts.begin(), clusterCounts.end());
	std::sort(clusterSizes.begin(), clusterSizes.end(),
		[](const auto& a, const auto& b) { return a.second > b.second; });

	int biggestCluster = clusterSizes[0].first;
	int leastCluster = clusterSizes[clusterSizes.size() - 1].first;

	if (clusterSizes[0].second == clusterSizes[clusterSizes.size() - 1].second) {
		return CMySmartPointer<CIndividual>(new CIndividual(fst));
	}

	std::random_device rd;
	std::mt19937 gen(rd());
	std::shuffle(fstCOPY->begin(), fstCOPY->end(), gen);

	int toSwap = this->numberOfClusters / 2;

	std::vector<Change> changes;
	changes.reserve(toSwap);

	int swapped = 0;
	for (int i = 0; i < fstCOPY->size() && swapped < toSwap; ++i) {
		if ((*fstCOPY)[i] == biggestCluster) {
			changes.push_back(Change{
				i,                // point index
				biggestCluster,   // old cluster
				leastCluster     // new cluster
				});

			(*fstCOPY)[i] = leastCluster;
			swapped++;
		}
	}

	std::pair<double, CMySmartPointer<Matrix>> result =
		this->updateVIGMatrix(
			fst.getMatrix(),    
			*fstCOPY,          
			changes,          
			fst.getFitness()   
		);

	CIndividual* mutatedIndividual = new CIndividual(*fstCOPY, *(result.second));
	mutatedIndividual->setFitness(result.first);

	return CMySmartPointer<CIndividual>(mutatedIndividual);
}

CMySmartPointer<CIndividual> COptimizer::methodIIMutationAlgorithm(CIndividual& fs) {
	std::vector<int> fst(fs.getGenotype());

	std::random_device rd;
	std::default_random_engine generator(rd());
	std::uniform_int_distribution<int> indexDist(0, fst.size() - 1);
	std::uniform_int_distribution<int> clusterDist(1, this->numberOfClusters);

	int mutationSize = std::max(1, static_cast<int>(fst.size() * 0.1));

	std::vector<Change> changes;
	changes.reserve(mutationSize);  // Reserve space for efficiency

	for (int i = 0; i < mutationSize; ++i) {
		int idx = indexDist(generator);
		int oldCluster = fst[idx];
		int newCluster;

		do {
			newCluster = clusterDist(generator);
		} while (newCluster == oldCluster);

		changes.push_back(Change{
			idx,         // point index
			oldCluster,  // old cluster
			newCluster   // new cluster
			});

		fst[idx] = newCluster;
	}

	std::pair<double, CMySmartPointer<Matrix>> result =
		this->updateVIGMatrix(
			fs.getMatrix(),    
			fst,              
			changes,         
			fs.getFitness()  
		);

	CIndividual* mutatedIndividual = new CIndividual(fst, *(result.second));
	mutatedIndividual->setFitness(result.first);

	return CMySmartPointer<CIndividual>(mutatedIndividual);
}


/*
CMySmartPointer<CIndividual> COptimizer::methodILocalAlgorithm(CIndividual& fs) {
	std::vector<int> fst = fs.getGenotype();
	double bestFitness = fs.getFitness();
	CMySmartPointer<std::vector<int>> bestVector(new std::vector<int>(fst));
	std::vector<int> idxs;
	for (size_t idx = 0; idx < fst.size(); ++idx) {
		for (int i = 1; i <= this->c_evaluator.iGetUpperBound(); ++i) {
			fst[idx] = i; // Zmieniamy wartoœæ w danym indeksie
			idxs.push_back(idx);
			auto p1 = this->updateVIGMatrix(fs.getMatrix(), fst, idxs, bestFitness);
			if (p1.first < bestFitness) {
				bestFitness = p1.first;
				*bestVector = fst; // Aktualizujemy najlepszy wektor
				CIndividual* c = new CIndividual(*bestVector, *p1.second);
				c->setFitness(p1.first);
				return CMySmartPointer<CIndividual>(c);
			}
			fst[idx] = fs.getGenotype()[idx]; // Przywracamy pierwotn¹ wartoœæ
			idxs.clear();
		}
	}

	return CMySmartPointer<CIndividual>( new CIndividual(fs)); // Zwracamy najlepsze (lub oryginalne) rozwi¹zanie
}*/
CMySmartPointer<CIndividual> COptimizer::methodILocalAlgorithm(CIndividual& fs) {
	std::vector<int> currentSolution = fs.getGenotype();
	double currentFitness = fs.getFitness();
	double bestFitness = currentFitness;

	CMySmartPointer<std::vector<int>> bestSolution(new std::vector<int>(currentSolution));
	CMySmartPointer<Matrix> bestMatrix(new Matrix(fs.getMatrix()));

	for (size_t idx = 0; idx < currentSolution.size(); ++idx) {
		int originalCluster = currentSolution[idx];

		for (int newCluster = 1; newCluster <= this->c_evaluator.iGetUpperBound(); ++newCluster) {
			if (newCluster == originalCluster) continue; // Pomijamy aktualny klaster

			std::vector<Change> changes;
			changes.push_back(Change{
				static_cast<int>(idx),    
				originalCluster,          
				newCluster               
				});

			std::pair<double, CMySmartPointer<Matrix>> result =
				updateVIGMatrix(
					fs.getMatrix(),          
					currentSolution,         
					changes,                 
					currentFitness          
				);

			double newFitness = result.first;
			CMySmartPointer<Matrix> newVIG = result.second;

			if (newFitness < bestFitness) {
				bestFitness = newFitness;
				*bestSolution = currentSolution;
				*bestMatrix = *newVIG;

				CIndividual* improvedIndividual = new CIndividual(*bestSolution, *bestMatrix);
				improvedIndividual->setFitness(bestFitness);
				return CMySmartPointer<CIndividual>(improvedIndividual);
			}

			currentSolution[idx] = originalCluster;
		}
	}

	return CMySmartPointer<CIndividual>(new CIndividual(fs));
}




std::vector<CMySmartPointer<CIndividual>> COptimizer::susSelectionAlgorithm(std::vector<CIndividual>& population, int toNextGeneration) {
	auto populationSize = population.size();
	std::vector<double> selectionFitness(populationSize);

	double sumSelectionFitness = 0.0;
	for (auto i = 0; i < populationSize; ++i) {
		double fitness = 1.0 / (1.0 + betterdEvaluate(population[i].getGenotype(), this->c_evaluator.vGetPoints()));
		selectionFitness[i] = fitness;
		sumSelectionFitness += fitness;
	}

	for (double& fitness : selectionFitness) {
		fitness /= sumSelectionFitness;
	}

	std::vector<double> cumulativeProbabilities(populationSize);
	double cumulativeSum = 0.0;
	for (auto i = 0; i < populationSize; ++i) {
		cumulativeSum += selectionFitness[i];
		cumulativeProbabilities[i] = cumulativeSum;
	}

	if (toNextGeneration % 2 != 0) {
		toNextGeneration -= 1;
	}
	if (toNextGeneration == 0) toNextGeneration = 1;

	std::random_device rd;
	std::default_random_engine generator(rd());
	std::uniform_real_distribution<double> distribution(0.0, 1.0);

	double startPoint = distribution(generator);
	double spacing = 1.0 / toNextGeneration;

	std::vector<CMySmartPointer<CIndividual>> parents;
	for (int i = 0; i < toNextGeneration; ++i) {
		double target = std::fmod(startPoint + i * spacing, 1.0);
		auto it = std::lower_bound(cumulativeProbabilities.begin(), cumulativeProbabilities.end(), target);
		auto index = std::distance(cumulativeProbabilities.begin(), it);
		parents.push_back(CMySmartPointer<CIndividual>(new CIndividual(population[index])));
	}
	return parents;
}


CMySmartPointer<CIndividual> NGroupingChallenge::COptimizer::methodPXCrossoverAlgorithm(CIndividual& fst, CIndividual& snd) {
	std::vector<int> of1(fst.getGenotype());
	std::vector<int> of2(snd.getGenotype());

	if (of1.size() != of2.size() || of1.size() < 2) {
	}

	int size = static_cast<int>(of1.size());
	int point1 = rand() % size;
	int point2 = rand() % size;

	if (point1 > point2) {
		std::swap(point1, point2);
	}

	for (int i = point1; i <= point2; ++i) {
		std::swap(of1[i], of2[i]);
	}
	CMySmartPointer<CIndividual> c1(new CIndividual(of1, *(createVIG(this->c_evaluator.vGetPoints(), of1))));
	c1->setFitness(betterdEvaluate(of1, this->c_evaluator.vGetPoints()));
	CMySmartPointer<CIndividual> c2(new CIndividual(of2, *(createVIG(this->c_evaluator.vGetPoints(), of2))));
	c2->setFitness(betterdEvaluate(of2, this->c_evaluator.vGetPoints()));
	return (c1->getFitness() < c2->getFitness()) ? c1 : c2;
}



double NGroupingChallenge::COptimizer::betterdEvaluate(const int* piSolution, const std::vector<CPoint>& v_points) const {
	if (!piSolution || v_points.empty()) {
		return -1;
	}

	double d_distance_sum = 0;

	for (size_t i = 0; i < v_points.size(); i++) {
		for (size_t j = i + 1; j < v_points.size(); j++) {
			if (piSolution[i] == piSolution[j]) { // Only calculate for points in the same group
				double d_distance = v_points[i].dCalculateDistance(v_points[j]);

				if (d_distance < 0) { // Check for invalid distance
					return -1;
				}

				d_distance_sum += 2.0 * d_distance; // Add the distance twice
			}
		}
	}

	return d_distance_sum;
}

double NGroupingChallenge::COptimizer::betterdEvaluate(const std::vector<int>* pvSolution, const std::vector<CPoint>& v_points) const {
	if (!pvSolution || pvSolution->size() != v_points.size()) {
		return -1;
	}

	return betterdEvaluate(pvSolution->data(), v_points);
}

double NGroupingChallenge::COptimizer::betterdEvaluate(const std::vector<int>& vSolution, const std::vector<CPoint>& v_points) const {
	if (vSolution.size() != v_points.size()) {
		return -1;
	}

	return betterdEvaluate(vSolution.data(), v_points);
}




double computeVIGValue(int clusterA, int clusterB, double distance) {
	if (clusterA == clusterB) {
		return 1.0;
	}
	return 0;
}

double COptimizer::computeFitnessChange(int idx, int oldCluster, int newCluster,
	const std::vector<int>& c, const Matrix& vig) {
	double change = 0.0;
	

	for (size_t j = 0; j < c.size(); ++j) {
		if (j == idx) continue;

		double distance = this->c_evaluator.vGetPoints()[idx].dCalculateDistance(this->c_evaluator.vGetPoints()[j]);
		double interaction = vig.getElem(idx, j);

		if (c[j] == oldCluster) {
			change -= interaction * distance;
		}
		if (c[j] == newCluster) {
			change += interaction * distance;
		}
	}
	return change;
}


std::pair<double, CMySmartPointer<Matrix>> COptimizer::updateVIGMatrix(
	const Matrix& oldVIG,
	std::vector<int>& c,
	const std::vector<Change>& changes,
	double oldFitness) {

	int n = oldVIG.getVSize();
	CMySmartPointer<Matrix> newVIG(new Matrix(n, n));

	// Kopiuj star¹ macierz
	for (int i = 0; i < n; i++) {
		for (int j = 0; j < n; j++) {
			newVIG->setOn(i, j, oldVIG.getElem(i, j));
		}
	}


	double newFitness = oldFitness;
	

	// Przetwórz ka¿d¹ zmianê
	for (const auto& change : changes) {
		int idx = change.pointIdx;
		int oldCluster = change.oldCluster;
		int newCluster = change.newCluster;

		// Aktualizuj VIG dla zmienionego punktu
		for (int j = 0; j < n; j++) {
			if (j == idx) {
				newVIG->setOn(idx, idx, 0);  // Przek¹tna
				continue;
			}

			double distance = this->c_evaluator.vGetPoints()[idx].dCalculateDistance(this->c_evaluator.vGetPoints()[j]);

			int vigValue;
			if (c[j] == newCluster) {
				vigValue = 1;
			}
			else {
				vigValue = 0;
			}

			// Aktualizuj symetrycznie
			newVIG->setOn(idx, j, vigValue);
			newVIG->setOn(j, idx, vigValue);
		}

		double fitnessChange = 0.0;
		for (int j = 0; j < n; j++) {
			if (j == idx) continue;

			double distance = this->c_evaluator.vGetPoints()[idx].dCalculateDistance(this->c_evaluator.vGetPoints()[j]);

			if (c[j] == oldCluster) {
				fitnessChange -= (oldVIG.getElem(idx, j)) * distance;
			}
			if (c[j] == newCluster) {
				fitnessChange += (newVIG->getElem(idx, j)) * distance;
			}
		}

		newFitness += fitnessChange;
		c[idx] = newCluster;
	}

	return { newFitness, newVIG };
}