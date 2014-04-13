#include "Indice2D.h"
#include "FonctionsRippling.h"

/*----------------------------------------------------------------------*\
 |*			Declaration 					*|
 \*---------------------------------------------------------------------*/

/*--------------------------------------*\
 |*		Public			*|
 \*-------------------------------------*/
__global__ void kernelFillImage(uchar4* ptrDevImageGL, int w, int h, float t);

/*--------------------------------------*\
 |*		Private			*|
 \*-------------------------------------*/
__device__ static void setPixel(uchar4& pixel, int i, int j, int w, int h, float t);

/*----------------------------------------------------------------------*\
 |*			Implementation 					*|
 \*---------------------------------------------------------------------*/

/*--------------------------------------*\
 |*		Public			*|
 \*-------------------------------------*/
void launchKernelFillImage(uchar4* ptrDevImageGL, int w, int h, float t)
    {
    dim3 dg = dim3(8, 8);
    dim3 db = dim3(16, 16, 1);
kernelFillImage<<<dg,db>>>(ptrDevImageGL,w,h,t);
}

__global__ void kernelFillImage(uchar4* ptrDevImageGL, int w, int h, float t)
{
const int n = w * h;
int tid = Indice2D::tid();
const int NB_THREAD = Indice2D::nbThread();
int s = tid;
int i;
int j;

while (s < n)
    {
    Indice2D::pixelIJ(s, w, i, j);
    setPixel(ptrDevImageGL[s], i, j, w, h, t);
    s += NB_THREAD;
    }
}

/*--------------------------------------*\
 |*		Private			*|
 \*-------------------------------------*/
__device__ void setPixel(uchar4& pixelIJ, int i, int j, int w, int h, float t)
{
unsigned char niveauGris = levelGrey(i, j, w, h, t);

pixelIJ.x = niveauGris;
pixelIJ.y = niveauGris;
pixelIJ.z = niveauGris;
pixelIJ.w = 255;
}

/*----------------------------------------------------------------------*\
 |*			End	 					*|
 \*---------------------------------------------------------------------*/
