#include "RipplingImageCudaMOOs.h"

/*----------------------------------------------------------------------*\
 |*			Declaration 					*|
 \*---------------------------------------------------------------------*/

/*--------------------------------------*\
 |*		Public			*|
 \*-------------------------------------*/
extern void launchKernelFillImage(uchar4* ptrDevImageGL, int w, int h, float t);

/*----------------------------------------------------------------------*\
 |*			Implementation					*|
 \*---------------------------------------------------------------------*/

/*--------------------------------------*\
 |*		Public			*|
 \*-------------------------------------*/
RipplingImageCudaMOOs::RipplingImageCudaMOOs(unsigned int w, unsigned int h, float tStart, float dt) :
	ImageCudaMOOs_A(w, h)
    {
    this->t = tStart;
    this->dt = dt;
    }

void RipplingImageCudaMOOs::animationStep(bool& isNeedUpdateView)
    {
    t += dt;
    isNeedUpdateView = true;
    }

void RipplingImageCudaMOOs::fillImageGL(uchar4* ptrDevImageGL, int w, int h)
    {
    launchKernelFillImage(ptrDevImageGL, w, h, t);
    }

RipplingImageCudaMOOs::~RipplingImageCudaMOOs()
    {

    }

/*----------------------------------------------------------------------*\
 |*			End	 					*|
 \*---------------------------------------------------------------------*/

