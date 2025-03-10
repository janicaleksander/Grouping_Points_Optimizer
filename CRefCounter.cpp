#include "CRefCounter.h"

CRefCounter::CRefCounter() : i_count(0) {}

int CRefCounter::iAdd() {
	return ++this->i_count;
}

int CRefCounter::iDec() {
	return --this->i_count;
}

int CRefCounter::iGet() {
	return this->i_count;
}


