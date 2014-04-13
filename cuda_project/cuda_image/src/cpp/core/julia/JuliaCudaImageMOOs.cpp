#include "JuliaCudaImageMOOs.h"
#include <cmath>
#include <iostream>

using namespace std;

/*----------------------------------------------------------------------*\
 |*			Declaration 					*|
 \*---------------------------------------------------------------------*/

/*--------------------------------------*\
 |*		Public			*|
 \*-------------------------------------*/
extern void launchKernelFillImageJulia(uchar4* ptrDevImageGL, int w, int h, float t, const DomaineMaths& domaineMath, int N, float* c);

/*----------------------------------------------------------------------*\
 |*			Implementation 					*|
 \*---------------------------------------------------------------------*/

/*--------------------------------------*\
 |*		Public			*|
 \*-------------------------------------*/
JuliaCudaImageMOOs::JuliaCudaImageMOOs(unsigned int w, unsigned int h, DomaineMaths domainMath, float tStart, float dt, int N, float* c) :
	ImageCudaFonctionelMOOs_A(w, h, domainMath)
    {
    this->t = t;
    this->dt = tStart;
    this->N = N;
    this->c = c;
    }

void JuliaCudaImageMOOs::fillImageGL(uchar4* ptrDevImageGL, int w, int h, const DomaineMaths& domaineMath)
    {
    launchKernelFillImageJulia(ptrDevImageGL, w, h, this->t, domaineMath, this->N, this->c);
    }

void JuliaCudaImageMOOs::animationStep(bool& isNeedUpdate)
    {
    this->t += dt;
//    this->N++;
    isNeedUpdate = true;
    }

JuliaCudaImageMOOs::~JuliaCudaImageMOOs()
    {
    if (c != 0)
	{
	delete[] c;
	}

    c = 0;
    }

/*----------------------------------------------------------------------*\
 |*			End	 					*|
 \*---------------------------------------------------------------------*/

