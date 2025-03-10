#include "Matrix.h"
#include <stdexcept>
#include <iostream>

Matrix::Matrix() : m_width(0), m_height(0), m(nullptr) {}

Matrix::Matrix(int x, int y) : m_width(x), m_height(y) {
    m = new int[x * y]();  
}

Matrix::Matrix(const Matrix& other) : m_width(other.m_width), m_height(other.m_height) {
    m = new int[m_width * m_height];  
    std::copy(other.m, other.m + m_width * m_height, m); 
}

Matrix& Matrix::operator=(const Matrix& other) {
    if (this != &other) {
        delete[] m; 
        m_width = other.m_width;
        m_height = other.m_height;
        m = new int[m_width * m_height]; 
        std::copy(other.m, other.m + m_width * m_height, m); 
    }
    return *this;
}

Matrix& Matrix::operator=(Matrix&& other) noexcept {
    if (this != &other) {
        delete[] m; 
        m_width = other.m_width;
        m_height = other.m_height;
        m = other.m; 
        other.m = nullptr;
    }
    return *this;
}

Matrix::Matrix(Matrix&& other) noexcept : m_width(other.m_width), m_height(other.m_height), m(other.m) {
    other.m = nullptr;
}

int Matrix::index(int x, int y) const {
    if (x < 0 || x >= m_width || y < 0 || y >= m_height) {
    }
    return x + m_width * y;
}

int Matrix::getElem(int posX, int posY) const {
    return m[index(posX, posY)];
}

void Matrix::setOn(int posX, int posY, int elem) {
    m[index(posX, posY)] = elem;
}

int Matrix::getVSize() const {
    return m_width;
}

Matrix::~Matrix() {
    delete[] m; 
}
