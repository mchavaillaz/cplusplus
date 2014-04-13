#ifndef HISTOGRAMMEHOST_H_
#define HISTOGRAMMEHOST_H_

#include "curand_kernel.h"

/*----------------------------------------------------------------------*\
 |*			Declaration 					*|
 \*---------------------------------------------------------------------*/

/*--------------------------------------*\
 |*		Public			*|
 \*-------------------------------------*/
class Histogramme
    {
    public:
	Histogramme();
	void run();
	virtual ~Histogramme();

	void printTabFrequence();

    private:
	//Output
	int* tabFrequence;

	//Tools
	int sizeArray;
	int sizeBlock;
	int deviceId;
	int n;
	dim3 dg;
	dim3 db;
	size_t sizeTabGenerator;
	size_t sizeSharedMemory;
	size_t sizePtrDevTabFrequence;

	int* ptrDevTabFrequence;
	curandState* ptrDevTabGeneratorThread;

	void runBuildGeneratorHistogramme();
	void runComputeHistogramme();
    };

#endif 

/*----------------------------------------------------------------------*\
 |*			End	 					*|
 \*---------------------------------------------------------------------*/
