#ifndef MONTECARLO_H_
#define MONTECARLO_H_

#include "curand_kernel.h"

/*----------------------------------------------------------------------*\
 |*			Declaration 					*|
 \*---------------------------------------------------------------------*/

/*--------------------------------------*\
 |*		Public			*|
 \*-------------------------------------*/
class MonteCarlo
    {
    public:
	MonteCarlo(float _a, float b, float _m, int _n);
	void run();
	virtual ~MonteCarlo();

	float getResultat();

    private:
	//Input
	float a;
	float b;
	float m;
	int n;

	//Output
	long sommeInTot;

	//Tools
	int deviceId;
	long sizeTabSM;
	dim3 dg;
	dim3 db;
	size_t sizeResultat;
	size_t sizeTabGenerator;
	size_t sizeShareMemory;

	long* ptrDevSommeInTot;
	curandState* ptrDevTabGeneratorThread;

	void runBuildGenerator();
	void runComputePI();
    };

#endif 

/*----------------------------------------------------------------------*\
 |*			End	 					*|
 \*---------------------------------------------------------------------*/
