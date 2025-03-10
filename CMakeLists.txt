cmake_minimum_required(VERSION 3.10)
project(ClusteringAlgorithm CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)

# Source files
set(SOURCES
        Matrix.cpp
        Optimizer.cpp
        Point.cpp
        CIndividual.cpp
        CRefCounter.cpp
        GaussianGroupingEvaluatorFactory.cpp
        GroupingChallenge.cpp
        GroupingEvaluator.cpp
)

# Header files
set(HEADERS
        Matrix.h
        Optimizer.h
        Point.h
        CIndividual.h
        CMySmartPointer.h
        CRefCounter.h
        GaussianGroupingEvaluatorFactory.h
        GroupingEvaluator.h
)

add_executable(clustering ${SOURCES} ${HEADERS})

# Include directories
include_directories(${CMAKE_SOURCE_DIR})
