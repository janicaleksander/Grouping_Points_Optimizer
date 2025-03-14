#include "GaussianGroupingEvaluatorFactory.h"
#include "GroupingEvaluator.h"
#include "Optimizer.h"

using namespace NGroupingChallenge;

int main(){
	CGaussianGroupingEvaluatorFactory c_evaluator_factory(5, 200, 5);

	c_evaluator_factory
		.cAddDimension(10, 100, 1.0, 1.0)
		.cAddDimension(10, 100, 1.0, 1.0)
		.cAddDimension(10, 100, 1.0, 1.0)
		.cAddDimension(10, 100, 1.0, 1.0)
		.cAddDimension(10, 100, 1.0, 1.0)
		.cAddDimension(10, 100, 1.0, 1.0)
		.cAddDimension(10, 100, 1.0, 1.0)
		.cAddDimension(10, 100, 1.0, 1.0)
		.cAddDimension(10, 100, 1.0, 1.0)
		.cAddDimension(10, 100, 1.0, 1.0);

	CGroupingEvaluator* pc_evaluator = c_evaluator_factory.pcCreateEvaluator(0);

	COptimizer c_optimizer(*pc_evaluator);

	c_optimizer.vInitialize();

	for (int i = 0; i < 10; i++){
		c_optimizer.vRunIteration();
	}

	delete pc_evaluator;

	return 0;
}