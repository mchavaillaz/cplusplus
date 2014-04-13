#include "MonteCarloHost.h"
#include "cuda.h"
#include "cudaTools.h"
#include "Device.h"
#include <iostream>
#include "Lock.h"
#include "MonteCarloHost.h"

using namespace ::std;

/*----------------------------------------------------------------------*\
 |*			Declaration 					*|
 \*---------------------------------------------------------------------*/
extern void __global__ kernelMonteCarloRand(curandState* ptrDevTabGeneratorThread, int deviceId, int n);
extern void __global__ kernelMonteCarloWork(curandState* ptrDevTabGeneratorThread, long* ptrDevSommeInTot, float a, float b, float m, Lock lock, int n);

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
MonteCarlo::MonteCarlo(float _a, float _b, float _m, int _n)
    {
    //Input
    this->a = _a;
    this->b = _b;
    this->m = _m;
    this->n = _n;

    //Output
    this->sommeInTot = 0.0;

    //Tools
    this->deviceId = Device::getDeviceId();
    int sizeParBlock = n / 16;
    this->dg = dim3(16, 1, 1);
    this->db = dim3(sizeParBlock, 1, 1);

    //Device
    this->sizeTabGenerator = sizeof(curandState) * n;
    this->sizeShareMemory = sizeof(long) * sizeParBlock;
    this->sizeResultat = sizeof(long);

    HANDLE_ERROR(cudaMalloc((void**) &ptrDevTabGeneratorThread, sizeTabGenerator));
    HANDLE_ERROR(cudaMalloc((void**) &ptrDevSommeInTot, sizeResultat));
    HANDLE_ERROR(cudaMemset(ptrDevSommeInTot, 0, sizeResultat));

    Device::assertDim(dg, db);
    }

MonteCarlo::~MonteCarlo()
    {
    HANDLE_ERROR(cudaFree(ptrDevSommeInTot));
    HANDLE_ERROR(cudaFree(ptrDevTabGeneratorThread));
    }

void MonteCarlo::run()
    {
    runBuildGenerator();
    runComputePI();
    }

void MonteCarlo::runBuildGenerator()
    {
    kernelMonteCarloRand<<<dg,db>>>(ptrDevTabGeneratorThread, deviceId, n);
    check_CUDA_Error("kernel rand");
    HANDLE_ERROR(cudaDeviceSynchronize());
    }

void MonteCarlo::runComputePI()
    {
    Lock lock;

    kernelMonteCarloWork<<<dg,db, this->sizeShareMemory>>>(ptrDevTabGeneratorThread, ptrDevSommeInTot, a, b, m, lock, n);
    check_CUDA_Error("kernel work");

    //Device -> Host
    HANDLE_ERROR(cudaMemcpy(&sommeInTot, ptrDevSommeInTot, this->sizeResultat, cudaMemcpyDeviceToHost));
    }

float MonteCarlo::getResultat()
    {
    float numerateur = (this->b - this->a) * this->m * this->sommeInTot;
    float denominateur = this->n*10000;

    return (numerateur/denominateur)/2;
    }

/*--------------------------------------*\
 |*		Private			*|
 \*-------------------------------------*/

/*----------------------------------------------------------------------*\
 |*			End	 					*|
 \*---------------------------------------------------------------------*/

