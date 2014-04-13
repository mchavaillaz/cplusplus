#include "NewtonCudaImageMOOs.h"
#include <cmath>
#include <iostream>

using namespace std;

/*----------------------------------------------------------------------*\
 |*			Declaration 					*|
 \*---------------------------------------------------------------------*/

/*--------------------------------------*\
 |*		Public			*|
 \*-------------------------------------*/
extern void launchKernelFillImageNewton(uchar4* ptrDevImageGL, int w, int h, float t, const DomaineMaths& domaineMath, int N);

/*----------------------------------------------------------------------*\
 |*			Implementation 					*|
 \*---------------------------------------------------------------------*/

/*--------------------------------------*\
 |*		Public			*|
 \*-------------------------------------*/
NewtonCudaImageMOOs::NewtonCudaImageMOOs(unsigned int w, unsigned int h, unsigned int N, DomaineMaths domainMath, float tStart, float dt) :
	ImageCudaFonctionelMOOs_A(w, h, domainMath)
    {
    this->t = t;
    this->dt = tStart;
    this->N = N;
    }

void NewtonCudaImageMOOs::fillImageGL(uchar4* ptrDevImageGL, int w, int h, const DomaineMaths& domaineMath)
    {
    launchKernelFillImageNewton(ptrDevImageGL, w, h, this->t, domaineMath, this->N);
    }

void NewtonCudaImageMOOs::animationStep(bool& isNeedUpdate)
    {
    this->t += dt;
    this->N++;
    isNeedUpdate = true;
    }

NewtonCudaImageMOOs::~NewtonCudaImageMOOs()
    {
    }

/*----------------------------------------------------------------------*\
 |*			End	 					*|
 \*---------------------------------------------------------------------*/

