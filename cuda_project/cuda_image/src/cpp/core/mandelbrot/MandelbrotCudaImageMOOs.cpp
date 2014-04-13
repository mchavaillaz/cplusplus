#include "MandelbrotCudaImageMOOs.h"
#include <cmath>
#include <iostream>

using namespace std;

/*----------------------------------------------------------------------*\
 |*			Declaration 					*|
 \*---------------------------------------------------------------------*/

/*--------------------------------------*\
 |*		Public			*|
 \*-------------------------------------*/
extern void launchKernelFillImageMandelbrot(uchar4* ptrDevImageGL, int w, int h, float t, const DomaineMaths& domaineMath, int N);

/*----------------------------------------------------------------------*\
 |*			Implementation 					*|
 \*---------------------------------------------------------------------*/

/*--------------------------------------*\
 |*		Public			*|
 \*-------------------------------------*/
MandelbrotCudaImageMOOs::MandelbrotCudaImageMOOs(unsigned int w, unsigned int h, DomaineMaths domainMath, float tStart, float dt, int N) :
	ImageCudaFonctionelMOOs_A(w, h, domainMath)
    {
    this->t = t;
    this->dt = tStart;
    this->N = N;
    }

void MandelbrotCudaImageMOOs::fillImageGL(uchar4* ptrDevImageGL, int w, int h, const DomaineMaths& domaineMath)
    {
    launchKernelFillImageMandelbrot(ptrDevImageGL, w, h, this->t, domaineMath, this->N);
    }

void MandelbrotCudaImageMOOs::animationStep(bool& isNeedUpdate)
    {
    this->t += dt;
    this->N++;
    isNeedUpdate = true;
    }

MandelbrotCudaImageMOOs::~MandelbrotCudaImageMOOs()
    {
    }

/*----------------------------------------------------------------------*\
 |*			End	 					*|
 \*---------------------------------------------------------------------*/

