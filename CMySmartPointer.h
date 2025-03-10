#pragma once
#include "CRefCounter.h"
#include <utility> // for std::exchange

template <typename T>
class CMySmartPointer {
public:
    CMySmartPointer() noexcept : pc_pointer(nullptr), ref_counter(nullptr) {}
    explicit CMySmartPointer(T* pcPointer) noexcept;
    ~CMySmartPointer();

    CMySmartPointer(const CMySmartPointer& cOther) noexcept;            
    CMySmartPointer(CMySmartPointer&& cOther) noexcept;                 
    CMySmartPointer& operator=(const CMySmartPointer& cOther) noexcept; 
    CMySmartPointer& operator=(CMySmartPointer&& cOther) noexcept;      

    T& operator*() const noexcept;
    T* operator->() const noexcept;
    explicit operator bool() const noexcept; 

    void reset(T* newPointer = nullptr) noexcept; 
    T* get() const noexcept;                     

private:
    T* pc_pointer;
    CRefCounter* ref_counter;

    void release() noexcept; 
};

// Constructor
template<typename T>
inline CMySmartPointer<T>::CMySmartPointer(T* pcPointer) noexcept
    : pc_pointer(pcPointer), ref_counter(pcPointer ? new CRefCounter() : nullptr) {
    if (ref_counter) {
        ref_counter->iAdd();
    }
}

// Destructor
template<typename T>
inline CMySmartPointer<T>::~CMySmartPointer() {
    release();
}

// Copy constructor
template<typename T>
inline CMySmartPointer<T>::CMySmartPointer(const CMySmartPointer& cOther) noexcept
    : pc_pointer(cOther.pc_pointer), ref_counter(cOther.ref_counter) {
    if (ref_counter) {
        ref_counter->iAdd();
    }
}

// Move constructor
template<typename T>
inline CMySmartPointer<T>::CMySmartPointer(CMySmartPointer&& cOther) noexcept
    : pc_pointer(std::exchange(cOther.pc_pointer, nullptr)),
    ref_counter(std::exchange(cOther.ref_counter, nullptr)) {
}

// Copy assignment operator
template<typename T>
inline CMySmartPointer<T>& CMySmartPointer<T>::operator=(const CMySmartPointer& cOther) noexcept {
    if (this != &cOther) {
        release();
        pc_pointer = cOther.pc_pointer;
        ref_counter = cOther.ref_counter;
        if (ref_counter) {
            ref_counter->iAdd();
        }
    }
    return *this;
}

// Move assignment operator
template<typename T>
inline CMySmartPointer<T>& CMySmartPointer<T>::operator=(CMySmartPointer&& cOther) noexcept {
    if (this != &cOther) {
        release();
        pc_pointer = std::exchange(cOther.pc_pointer, nullptr);
        ref_counter = std::exchange(cOther.ref_counter, nullptr);
    }
    return *this;
}

// Release memory
template<typename T>
inline void CMySmartPointer<T>::release() noexcept {
    if (ref_counter && ref_counter->iDec() == 0) {
        delete pc_pointer;
        delete ref_counter;
    }
    pc_pointer = nullptr;
    ref_counter = nullptr;
}

// Dereference operators
template<typename T>
inline T& CMySmartPointer<T>::operator*() const noexcept {
    return *pc_pointer;
}

template<typename T>
inline T* CMySmartPointer<T>::operator->() const noexcept {
    return pc_pointer;
}

// Bool conversion for null-checks
template<typename T>
inline CMySmartPointer<T>::operator bool() const noexcept {
    return pc_pointer != nullptr;
}

// Reset smart pointer to a new raw pointer
template<typename T>
inline void CMySmartPointer<T>::reset(T* newPointer) noexcept {
    release();
    if (newPointer) {
        pc_pointer = newPointer;
        ref_counter = new CRefCounter();
        ref_counter->iAdd();
    }
    else {
        pc_pointer = nullptr;
        ref_counter = nullptr;
    }
}

// Get raw pointer
template<typename T>
inline T* CMySmartPointer<T>::get() const noexcept {
    return pc_pointer;
}
