#pragma once
class Matrix
{
public:

    Matrix();  
    ~Matrix();
    Matrix(int x, int y);  
    Matrix(const Matrix& other); 
    Matrix& operator=(const Matrix& other); 
    Matrix& operator=(Matrix&& other) noexcept; 
    Matrix(Matrix&& other) noexcept;  

    int getElem(int posX, int posY) const; 
    void setOn(int posX, int posY, int elem);  
    int getVSize() const; 

private:
    int m_width;
    int m_height;
    int* m; 

    int index(int x, int y) const; 
};
