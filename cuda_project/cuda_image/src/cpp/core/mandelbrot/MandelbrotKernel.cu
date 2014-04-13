#include "Indice2D.h"
#include "FonctionsMandelbrot.h"
#include "DomaineMaths.h"
#include "DomaineMathGPUs.h"
#include "IndiceXY.h"
#include "cudaTools.h"
#include "ColorToolCuda.h"

/*----------------------------------------------------------------------*\
 |*			Declaration 					*|
 \*---------------------------------------------------------------------*/

/*--------------------------------------*\
 |*		Public			*|
 \*-------------------------------------*/
__global__ void kernelFillImageMandelbrot(uchar4* ptrDevImageGL, int w, int h, DomaineMathGPUs domaineMathGPUs, float t, int N);

/*--------------------------------------*\
 |*		Private			*|
 \*-------------------------------------*/
__device__ static void setPixel(uchar4& pixel, int i, int j, int w, int h, float t, int N, IndiceXY indiceXY);

/*----------------------------------------------------------------------*\
 |*			Implementation 					*|
 \*---------------------------------------------------------------------*/

/*--------------------------------------*\
 |*		Public			*|
 \*-------------------------------------*/
void launchKernelFillImageMandelbrot(uchar4* ptrDevImageGL, int w, int h, float t, const DomaineMaths& domaineMath, int N)
    {
    dim3 dg = dim3(8, 8);
    dim3 db = dim3(16, 16, 1);

    DomaineMathGPUs domaineMathGPUs(domaineMath);

kernelFillImageMandelbrot<<<dg,db>>>(ptrDevImageGL, w, h, domaineMathGPUs, t, N);
}

__global__ void kernelFillImageMandelbrot(uchar4* ptrDevImageGL, int w, int h, DomaineMathGPUs domaineMathGPUs, float t, int N)
{
IndiceXY indiceXY(w, h, &domaineMathGPUs);
const int n = w * h;
int tid = Indice2D::tid();
const int NB_THREAD = Indice2D::nbThread();
int s = tid;
int i;
int j;

while (s < n)
    {
    Indice2D::pixelIJ(s, w, i, j);
    setPixel(ptrDevImageGL[s], i, j, w, h, t, N, indiceXY);
    ptrDevImageGL[s].w = 255;
    s += NB_THREAD;
    }
}

/*--------------------------------------*\
 |*		Private			*|
 \*-------------------------------------*/
__device__ void setPixel(uchar4& pixelIJ, int i, int j, int w, int h, float t, int N, IndiceXY indiceXY)
{
float x;
float y;
indiceXY.toXY(i, j, x, y);

int k = calculateSequence(x, y, N);
float hue = calculateHk(k, N);
float s = 1.0;
float b = 1.0;

if (k <= 0)
    {
    hue = 0.0;
    s = 0.0;
    b = 0.0;
    }

ColorToolCuda::HSB_TO_RVB(hue, s, b, pixelIJ.x, pixelIJ.y, pixelIJ.z);
}

/*----------------------------------------------------------------------*\
 |*			End	 					*|
 \*---------------------------------------------------------------------*/
