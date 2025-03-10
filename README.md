# Clustering Algorithm in C++

##  Overview
This project implements a clustering algorithm in C++ for grouping points based on their similarities. The algorithm does not rely on centroids but instead uses a vector representation where each point is assigned to a cluster. It is a implementation of genetic algorithm with local search and VIG optimization technique.

##  Features
- ✅ Efficient clustering of points
- ✅ No reliance on centroids
- ✅ Optimized for performance in C++

## ⚙️ Requirements
- 🖥️ C++20 or later
- 🛠️ CMake (for building the project)


## 🚀 Installation (Linux)
```sh
# Clone the repository
git clone https://github.com/janicaleksander/Grouping_Points_Optimizer.git

# Build the project using CMake
mkdir build 
cd build
cmake ..
make
```

## ▶️ Usage
Run the compiled binary. First you can setup start conditions. 
```
CGaussianGroupingEvaluatorFactory c_evaluator_factory(x, y, z);
x - number of cluster
y - number of points 
z - random seed

```
There you can see how your fitness function is decreasing.

## 🤝 Contributing
Contributions are welcome! Feel free to open an issue or submit a pull request.

## 📜 License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

