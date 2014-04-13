#include "Indice1D.h"
#include "Indice2D.h"
#include "cuda.h"
#include "curand_kernel.h"
#include "Lock.h"
#include "Device.h"
#include <cmath>
#include <stdio.h>
#include <iostream>
#include "cudaTools.h"
#include "FonctionsHistogramme.h"

/*----------------------------------------------------------------------*\
 |*			Declaration 					*|
 \*---------------------------------------------------------------------*/

/*--------------------------------------*\
 |*		Public			*|
 \*-------------------------------------*/
__global__ void kernelHistogrammeRand(curandState* ptrDevTabGeneratorThread, int deviceId, int n);
__global__ void kernelHistogrammeWork(curandState* ptrDevTabGeneratorThread, int* ptrDevTabFrequence, Lock lock, int n);
__device__ static void reductionInterBlockHistogramme(int* tabSM, Lock &lock, int* ptrDevTabFrequence, int n);
__device__ static void fillBlockHistogramme(curandState* ptrDevTabGeneratorThread, int* tabSM, int n);
__device__ static void initTabFrequenceHistogramme(int* ptrDevTabFrequence, Lock &lock, int n);
__device__ static void initTabSMHistogramme(int* tabSM, int n);

/*--------------------------------------*\
 |*		Private			*|
 \*-------------------------------------*/

/*----------------------------------------------------------------------*\
 |*			Implementation 					*|
 \*---------------------------------------------------------------------*/

/*--------------------------------------*\
 |*		Public			*|
 \*-------------------------------------*/
__global__ void kernelHistogrammeRand(curandState* ptrDevTabGeneratorThread, int deviceId)
    {
    int tid = Indice1D::tid();

    int deltaSeed = tid * INT_MAX;
    int deltaSequence = deviceId * 100;
    int deltaOffset = deviceId * 100;

    int seed = 1234 + deltaSeed;
    int sequenceNumber = tid + deltaSequence;
    int offset = deltaOffset;

    printf("rand tid=%d\n", tid);
    curand_init(seed, sequenceNumber, offset, &ptrDevTabGeneratorThread[tid]);
    }

__global__ void kernelHistogrammeWork(curandState* ptrDevTabGeneratorThread, int* ptrDevTabFrequence, Lock lock, int n)
    {
    extern __shared__ int tabSM[];
    int sizeTab = n/16;

    initTabFrequenceHistogramme(ptrDevTabFrequence, lock, sizeTab);
    __syncthreads();

    initTabSMHistogramme(tabSM, n);
    __syncthreads();

    fillBlockHistogramme(ptrDevTabGeneratorThread, tabSM, n);
    __syncthreads();

    reductionInterBlockHistogramme(tabSM, lock, ptrDevTabFrequence, n);
    __syncthreads();
    }

/*--------------------------------------*\
 |*		Private			*|
 \*-------------------------------------*/
__device__ void reductionInterBlockHistogramme(int* tabSM, Lock &lock, int* ptrDevTabFrequence, int n)
    {
    int tidLocal = threadIdx.x;

    if (tidLocal < n)
	{
	lock.lock();
	ptrDevTabFrequence[tidLocal] += tabSM[tidLocal];
	lock.unlock();
	}
    }

__device__ void fillBlockHistogramme(curandState* ptrDevTabGeneratorThread, int* tabSM, int n)
    {
    int tid = Indice1D::tid();
    curandState localState = ptrDevTabGeneratorThread[tid];

    float alea = 0.0;
    int valAlea = 0;

    for (int i = 0; i < 1; i++)
	{
	alea = curand_uniform(&localState);
	alea = convertNumberInRangeHistogramme(0, 2, alea);
	valAlea = alea;
	tabSM[valAlea] = tabSM[valAlea] + 1;
	}
    }

__device__ void initTabFrequenceHistogramme(int* ptrDevTabFrequence, Lock &lock, int n)
    {
    int tid = Indice1D::tid();

    if (tid < n)
	{
	lock.lock();
	ptrDevTabFrequence[tid] = 0;
	lock.unlock();
	}
    }

__device__ void initTabSMHistogramme(int* tabSM, int n)
    {
    int tidLocal = threadIdx.x;

    if (tidLocal < n)
	{
	tabSM[tidLocal] = 0;
	}
    }

/*----------------------------------------------------------------------*\
 |*			End	 					*|
 \*---------------------------------------------------------------------*/

