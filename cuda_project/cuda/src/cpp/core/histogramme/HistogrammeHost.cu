#include "cuda.h"
#include "cudaTools.h"
#include "Device.h"
#include <iostream>
#include "Lock.h"
#include "HistogrammeHost.h"

using namespace ::std;

/*----------------------------------------------------------------------*\
 |*			Declaration 					*|
 \*---------------------------------------------------------------------*/
extern void __global__ kernelHistogrammeRand(curandState* ptrDevTabGeneratorThread, int deviceId);
extern void __global__ kernelHistogrammeWork(curandState* ptrDevTabGeneratorThread, int* ptrDevTabFrequence, Lock lock, int n);

/*--------------------------------------*\
 |*		Public			*|
 \*-------------------------------------*/

/*--------------------------------------*\
 |*		Private			*|
 \*-------------------------------------*/

/*----------------------------------------------------------------------*\
 |*			Implementation 					*|
 \*---------------------------------------------------------------------*/

/*--------------------------------------*\
 |*		Public			*|
 \*-------------------------------------*/
Histogramme::Histogramme()
    {
    //Tools
    this->sizeArray = 256;
    this->sizeBlock = 16;
    this->n = this->sizeArray * this->sizeBlock;
    this->deviceId = Device::getDeviceId();
    this->dg = dim3(this->sizeBlock, 1, 1);
    this->db = dim3(this->sizeArray, 1, 1);

    //Output
    this->tabFrequence = new int[this->sizeArray];

    //Device
    this->sizeTabGenerator = sizeof(curandState) * this->n;
    this->sizeSharedMemory = sizeof(int) * this->sizeArray;
    this->sizePtrDevTabFrequence = sizeof(int) * this->sizeArray;

    HANDLE_ERROR(cudaMalloc((void**) &ptrDevTabGeneratorThread, this->sizeTabGenerator));
    HANDLE_ERROR(cudaMalloc((void**) &ptrDevTabFrequence, this->sizePtrDevTabFrequence));

    Device::assertDim(dg, db);
    }

Histogramme::~Histogramme()
    {
    HANDLE_ERROR(cudaFree(ptrDevTabGeneratorThread));
    HANDLE_ERROR(cudaFree(ptrDevTabFrequence));
    }

void Histogramme::run()
    {
    runBuildGeneratorHistogramme();
    runComputeHistogramme();
    }

void Histogramme::runBuildGeneratorHistogramme()
    {
    kernelHistogrammeRand<<<dg,db>>>(ptrDevTabGeneratorThread,deviceId);
    check_CUDA_Error("kernel rand");
    HANDLE_ERROR(cudaDeviceSynchronize());
    }

void Histogramme::runComputeHistogramme()
    {
    Lock lock;

    kernelHistogrammeWork<<<dg,db, this->sizeSharedMemory>>>(ptrDevTabGeneratorThread, ptrDevTabFrequence, lock, n);
    check_CUDA_Error("kernel work");

    //Device -> Host
    HANDLE_ERROR(cudaMemcpy(&tabFrequence, ptrDevTabFrequence, this->sizePtrDevTabFrequence, cudaMemcpyDeviceToHost));
    }

void Histogramme::printTabFrequence()
    {
    for(int i=0;i<this->sizeArray;i++)
	{
	cout << "i=" << i << " frequence=" << this->tabFrequence[i] << endl;
	}
    }

/*--------------------------------------*\
 |*		Private			*|
 \*-------------------------------------*/

/*----------------------------------------------------------------------*\
 |*			End	 					*|
 \*---------------------------------------------------------------------*/

