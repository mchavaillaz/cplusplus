#include "FonctionsMonteCarlo.h"
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

/*----------------------------------------------------------------------*\
 |*			Declaration 					*|
 \*---------------------------------------------------------------------*/

/*--------------------------------------*\
 |*		Public			*|
 \*-------------------------------------*/
__global__ void kernelMonteCarloRand(curandState* ptrDevTabGeneratorThread, int deviceId, int n);
__global__ void kernelMonteCarloWork(curandState* ptrDevTabGeneratorThread, long* ptrDevSommeInTot, float a, float b, float m, Lock lock, int n);
__device__ static void reductionIntraBlockMonteCarlo(long* tabSM, int n);
__device__ static void reductionInterBlockMonteCarlo(long* tabSM, Lock &lock, long* ptrDevResult);
__device__ static void ecrasementMonteCarlo(long* tabSM, int moitie);
__device__ static void fillBlockMonteCarlo(curandState* ptrDevTabGeneratorThread, long* tabSM, float a, float b, float m, int n);

/*--------------------------------------*\
 |*		Private			*|
 \*-------------------------------------*/

/*----------------------------------------------------------------------*\
 |*			Implementation 					*|
 \*---------------------------------------------------------------------*/

/*--------------------------------------*\
 |*		Public			*|
 \*-------------------------------------*/
__global__ void kernelMonteCarloRand(curandState* ptrDevTabGeneratorThread, int deviceId, int n)
    {
    int tid = Indice1D::tid();

    int deltaSeed = tid * INT_MAX;
    int deltaSequence = deviceId * 100;
    int deltaOffset = deviceId * 100;

    int seed = 1234 + deltaSeed;
    int sequenceNumber = tid + deltaSequence;
    int offset = deltaOffset;

    curand_init(seed, sequenceNumber, offset, &ptrDevTabGeneratorThread[tid]);
    }

__global__ void kernelMonteCarloWork(curandState* ptrDevTabGeneratorThread, long* ptrDevSommeInTot, float a, float b, float m, Lock lock, int n)
    {
    extern __shared__ long tabSM[];

    fillBlockMonteCarlo(ptrDevTabGeneratorThread, tabSM, a, b, m, n);
    __syncthreads();

    reductionIntraBlockMonteCarlo(tabSM, n);
    __syncthreads();

    reductionInterBlockMonteCarlo(tabSM, lock, ptrDevSommeInTot);
    __syncthreads();
    }

/*--------------------------------------*\
 |*		Private			*|
 \*-------------------------------------*/

__device__ void reductionIntraBlockMonteCarlo(long* tabSM, int n)
    {
    int moitie = n / 16;
    moitie = moitie / 2;

    while (moitie >= 1)
	{
	ecrasementMonteCarlo(tabSM, moitie);
	moitie = moitie / 2;
	__syncthreads();
	}
    }

__device__ void reductionInterBlockMonteCarlo(long* tabSM, Lock &lock, long* ptrDevSommeInTot)
    {
    int tidLocal = threadIdx.x;

    if (tidLocal == 0)
	{
	lock.lock();
	*ptrDevSommeInTot += tabSM[0];
	lock.unlock();
	}
    }

__device__ void ecrasementMonteCarlo(long* tabSM, int moitie)
    {
    int tidLocal = threadIdx.x;

    if (tidLocal < moitie)
	{
	tabSM[tidLocal] = tabSM[tidLocal] + tabSM[tidLocal + moitie];
	}
    }

__device__ void fillBlockMonteCarlo(curandState* ptrDevTabGeneratorThread, long* tabSM, float a, float b, float m, int n)
    {
    int tidLocal = threadIdx.x;
    int tid = Indice1D::tid();
    long sommeTot = 0;
    curandState localState = ptrDevTabGeneratorThread[tid];

    float xAlea = 0.0;
    float yAlea = 0.0;

    for (int i = 0; i < 10000; i++)
	{
	xAlea = curand_uniform(&localState);
	yAlea = curand_uniform(&localState);

	xAlea = convertNumberInRange(a, b, xAlea);
	yAlea = convertNumberInRange(0.0, m, yAlea);

	if (yAlea <= f(xAlea))
	    {
	    sommeTot += 1;
	    }
	}
    tabSM[tidLocal] = sommeTot;
    }

/*----------------------------------------------------------------------*\
 |*			End	 					*|
 \*---------------------------------------------------------------------*/

