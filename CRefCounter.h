#pragma once

class CRefCounter {
public:
	CRefCounter();
	int iAdd();
	int iDec();
	int iGet();
private:
	int i_count;
};