#include "cudaTools.h"
#include "Indice2D.h"
#include "Indice1D.h"
#include "Device.h"
#include "Lock.h"
#include "FonctionsProduitScalaire.h"
#include "cuda.h"
#include <cmath>
#include <cstdio>

/*----------------------------------------------------------------------*\
 |*			Declaration 					*|
 \*---------------------------------------------------------------------*/

/*--------------------------------------*\
 |*		Public			*|
 \*-------------------------------------*/
__global__ void kernelProduitScalaire(int n, Lock lock, float* ptrDevResult);
__device__ void reductionIntraBlock(float* tabSM, int n);
__device__ void reductionInterBlock(float* tabSM, Lock &lock, float* ptrDevResult);
__device__ void ecrasement(float* tabSM, int moitie);
__device__ void fillBlock(float* tabSM, int n);

/*--------------------------------------*\
 |*		Private			*|
 \*-------------------------------------*/

/*----------------------------------------------------------------------*\
 |*			Implementation 					*|
 \*---------------------------------------------------------------------*/

/*--------------------------------------*\
 |*		Public			*|
 \*-------------------------------------*/
void launchKernelProduitScalaire(int n, float &resultat)
    {
    int sizeTabPerBlock = n/16;
    dim3 dg = dim3(16, 1, 1);
    dim3 db = dim3(sizeTabPerBlock, 1, 1);

    Device::assertDim(dg, db);

    //Taille de tabSM en Shared Memory
    size_t size = sizeof(float) * sizeTabPerBlock;
    Lock lock;

    float *ptrDevResult;

    // Device memory allocation in GM
    HANDLE_ERROR(cudaMalloc((void**) &ptrDevResult, sizeof(float)));
    HANDLE_ERROR(cudaMemset(ptrDevResult, 0, sizeof(float)));

    kernelProduitScalaire<<<dg,db,size>>>(n/16, lock, ptrDevResult);

    // Device -> Host
    HANDLE_ERROR(cudaMemcpy(&resultat, ptrDevResult, sizeof(float), cudaMemcpyDeviceToHost));

    HANDLE_ERROR(cudaFree(ptrDevResult));
    }

__global__ void kernelProduitScalaire(int n, Lock lock, float *ptrDevResult)
    {
    extern __shared__ float tabSM[];

    fillBlock(tabSM, n);

    __syncthreads();

    reductionIntraBlock(tabSM, n);
    reductionInterBlock(tabSM, lock, ptrDevResult);

    __syncthreads();
    }

/*--------------------------------------*\
 |*		Private			*|
 \*-------------------------------------*/
__device__ void fillBlock(float* tabSM, int n)
    {
    int tidLocal = threadIdx.x;
    int i = Indice1D::tid();

    if (tidLocal < n)
	{
	tabSM[tidLocal] = v(i) * w(i);
	}
    }

__device__ void reductionIntraBlock(float* tabSM, int n)
    {
    int moitie = n / 2;

    while(moitie >= 1)
	{
	ecrasement(tabSM, moitie);
	moitie /= 2;
	__syncthreads();
	}
    }

__device__ void reductionInterBlock(float* tabSM, Lock &lock, float* ptrDevResult)
    {
    int tidLocal = threadIdx.x;

    if(tidLocal == 0)
	{
	lock.lock();
	*ptrDevResult += tabSM[0];
	lock.unlock();
	}
    }

__device__ void ecrasement(float* tabSM, int moitie)
    {
    int tidLocal = threadIdx.x;
    int i = tidLocal;

    if(i < moitie)
	{
	tabSM[i] = tabSM[i] + tabSM[i + moitie];
	}
    }

/*----------------------------------------------------------------------*\
 |*			End	 					*|
 \*---------------------------------------------------------------------*/
