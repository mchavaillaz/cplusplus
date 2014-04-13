#include "RaytracingCudaImageMOOsConstMemory.h"
#include <cstdlib>
#include <ctime>
#include <iostream>

using namespace std;

/*----------------------------------------------------------------------*\
 |*			Declaration 					*|
 \*---------------------------------------------------------------------*/

/*--------------------------------------*\
 |*		Public			*|
 \*-------------------------------------*/
extern void launchKernelFillImageRaytracingConstMemory(uchar4* ptrDevImageGL, int w, int h, float t, const DomaineMaths& domaineMath);
extern void allocMemoryConstMemory(unsigned int w, unsigned int h);
extern void freeMemoryConstMemory();

/*----------------------------------------------------------------------*\
 |*			Implementation 					*|
 \*---------------------------------------------------------------------*/

/*--------------------------------------*\
 |*		Public			*|
 \*-------------------------------------*/
RaytracingCudaImageMOOsConstMemory::RaytracingCudaImageMOOsConstMemory(unsigned int w, unsigned int h, DomaineMaths domainMath, float tStart, float dt) :
	ImageCudaFonctionelMOOs_A(w, h, domainMath)
    {
    this->t = tStart;
    this->dt = dt;

    allocMemoryConstMemory(w, h);
    }

void RaytracingCudaImageMOOsConstMemory::fillImageGL(uchar4* ptrDevImageGL, int w, int h, const DomaineMaths& domaineMath)
    {
    launchKernelFillImageRaytracingConstMemory(ptrDevImageGL, w, h, this->t, domaineMath);
    }

void RaytracingCudaImageMOOsConstMemory::animationStep(bool& isNeedUpdate)
    {
//    this->t += dt;
    isNeedUpdate = true;
    }

RaytracingCudaImageMOOsConstMemory::~RaytracingCudaImageMOOsConstMemory()
    {
    }

/*----------------------------------------------------------------------*\
 |*			End	 					*|
 \*---------------------------------------------------------------------*/

