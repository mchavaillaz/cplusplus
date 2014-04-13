#include "cudaTools.h"
#include "Indice2D.h"
#include "ColorToolCuda.h"
#include "CalibreurCudas.h"

/*----------------------------------------------------------------------*\
 |*			Declaration 					*|
 \*---------------------------------------------------------------------*/

/*--------------------------------------*\
 |*		Public			*|
 \*-------------------------------------*/
__global__ void kernelFillImageHeatTransfert(int w, int h, float* ptrDevImageA, float* ptrDevImageB, float* ptrDevImageHeaters, float* ptrDevImageInit);

__global__ void kernelInitImageHeatTransfert(int w, int h, float* ptrDevImageA, float* ptrDevImageB, float* ptrDevImageHeaters, float* ptrDevImageInit);

__global__ void kernelRenderImageHeatTransfert(uchar4* ptrDevImageGL, int w, int h, CalibreurCudas calibreur, float* ptrDevImageToRender);

/*--------------------------------------*\
 |*		Private			*|
 \*-------------------------------------*/
__device__ static void setPixel(uchar4& pixel, float valueOriginal);
__device__ static void ecrasement(float* ptrDevImageHeater, float* ptrDevImage, int tid);
__device__ static void diffusion(float* ptrDevImageDiffusion, float* ptrDevImageResult, int w, int h, int tid);
__device__ static void D(float* ptrDevImageDiffusion, float* ptrDevImageResult, int w, int h, int NB_THREAD, int n);
__device__ static void E(float* ptrDevImageHeater, float* ptrDevImage, int w, int h, int NB_THREAD, int n);

/*----------------------------------------------------------------------*\
 |*			Implementation 					*|
 \*---------------------------------------------------------------------*/

/*--------------------------------------*\
 |*		Public			*|
 \*-------------------------------------*/
void launchKernelFillImageHeatTransfert(int w, int h, float* ptrDevImageA, float* ptrDevImageB, float* ptrDevImageHeaters, float* ptrDevImageInit)
    {
    dim3 dg = dim3(8, 8);
    dim3 db = dim3(16, 16, 1);

    kernelFillImageHeatTransfert<<<dg,db>>>(w, h, ptrDevImageA, ptrDevImageB, ptrDevImageHeaters, ptrDevImageInit);
    int a = 0;
    }

void launchKernelInitImageHeatTransfert(int w, int h, float* ptrDevImageA, float* ptrDevImageB, float* ptrDevImageHeaters, float* ptrDevImageInit)
    {
    dim3 dg = dim3(8, 8);
    dim3 db = dim3(16, 16, 1);

    kernelInitImageHeatTransfert<<<dg,db>>>(w, h, ptrDevImageA, ptrDevImageB, ptrDevImageHeaters, ptrDevImageInit);
    int a = 0;
    }

void launchKernelRenderImageHeatTransfert(uchar4* ptrDevImageGL, int w, int h, CalibreurCudas calibreur, float* ptrDevImageToRender)
    {
    dim3 dg = dim3(8, 8);
    dim3 db = dim3(16, 16, 1);

    kernelRenderImageHeatTransfert<<<dg,db>>>(ptrDevImageGL, w, h, calibreur, ptrDevImageToRender);
    int a = 0;
    }

__global__ void kernelFillImageHeatTransfert(int w, int h, float* ptrDevImageA, float* ptrDevImageB, float* ptrDevImageHeaters, float* ptrDevImageInit)
    {
    const int n = w * h;
    const int NB_THREAD = Indice2D::nbThread();
    int s = Indice2D::tid();

    //ImageB = Diffusion(ImageA)
    D(ptrDevImageA, ptrDevImageB, w, h, NB_THREAD, n);

    //ImageB = Ecrasement(ImageHeaters, ImageB)
    E(ptrDevImageHeaters, ptrDevImageB, w, h, NB_THREAD, n);

    //ImageA = Diffusion(ImageB)
    D(ptrDevImageB, ptrDevImageA, w, h, NB_THREAD, n);

    //ImageA = Ecrasement(ImageHeaters, ImageA)
    E(ptrDevImageHeaters, ptrDevImageA, w, h, NB_THREAD, n);
    }

__global__ void kernelInitImageHeatTransfert(int w, int h, float* ptrDevImageA, float* ptrDevImageB, float* ptrDevImageHeaters, float* ptrDevImageInit)
    {
    const int n = w * h;
    const int NB_THREAD = Indice2D::nbThread();
    int s = Indice2D::tid();

    //ImageInit = Ecrasement(ImageHeaters, ImageInit)
    E(ptrDevImageHeaters, ptrDevImageInit, w, h, NB_THREAD, n);

    //ImageA = Diffusion(ImageInit)
    D(ptrDevImageInit, ptrDevImageA, w, h, NB_THREAD, n);

    //ImageA = Ecrasement(ImageHeaters, ImageA)
    E(ptrDevImageHeaters, ptrDevImageA, w, h, NB_THREAD, n);
    }

__global__ void kernelRenderImageHeatTransfert(uchar4* ptrDevImageGL, int w, int h, CalibreurCudas calibreur, float* ptrDevImageToRender)
    {
    const int n = w * h;
    const int NB_THREAD = Indice2D::nbThread();
    int s = Indice2D::tid();

    //Render ImageToRender
    while (s < n)
	{
	float hue = calibreur.calibrate(ptrDevImageToRender[s]);
	setPixel(ptrDevImageGL[s], hue);
	s += NB_THREAD;
	}

    __syncthreads();
    }

__device__ static void setPixel(uchar4& pixelIJ, float hue)
    {
    float h = hue;
    float s = 1.0;
    float b = 1.0;

    ColorToolCuda::HSB_TO_RVB(h, s, b, pixelIJ.x, pixelIJ.y, pixelIJ.z);
    pixelIJ.w = 255;
    }

__device__ static void ecrasement(float* ptrDevImageHeater, float* ptrDevImage, int tid)
    {
    float valueHeater = ptrDevImageHeater[tid];

    if (valueHeater != 0.0)
	{
	ptrDevImage[tid] = valueHeater;
	}
    }

__device__ static void diffusion(float* ptrDevImageDiffusion, float* ptrDevImageResult, int w, int h, int tid)
    {
    int size = w * h;
    float k = 0.25;
    int i;
    int j;
    float newValue = 0.0;
    float oldValue = ptrDevImageResult[tid];

    if (tid < size)
	{
	Indice2D::pixelIJ(tid, w, i, j);

	if ((i >= 3 && i < w-3) && (j >= 3 && j < h-3))
	    {
	    float westValue = ptrDevImageDiffusion[tid - 1];
	    float eastValue = ptrDevImageDiffusion[tid + 1];
	    float northValue = ptrDevImageDiffusion[tid - w];
	    float southValue = ptrDevImageDiffusion[tid + w];

	    newValue = oldValue + k * (westValue + eastValue + northValue + southValue - (4 * oldValue));
	    }
	else
	    {
	    newValue = oldValue;
	    }

	ptrDevImageResult[tid] = newValue;
	}

    }

__device__ static void D(float* ptrDevImageDiffusion, float* ptrDevImageResult, int w, int h, int NB_THREAD, int n)
    {
    int s = Indice2D::tid();

    //ptrDevImageResult = Diffusion(ptrDevImageDiffusion)
    while (s < n)
	{
	diffusion(ptrDevImageDiffusion, ptrDevImageResult, w, h, s);
	s += NB_THREAD;
	}

    __syncthreads();
    }

__device__ static void E(float* ptrDevImageHeaters, float* ptrDevImage, int w, int h, int NB_THREAD, int n)
    {
    int s = Indice2D::tid();

    //ptrDevImage = Ecrasement(ptrDevImageHeaters, ptrDevImage)
    while (s < n)
	{
	ecrasement(ptrDevImageHeaters, ptrDevImage, s);
	s += NB_THREAD;
	}

    __syncthreads();
    }

/*--------------------------------------*\
 |*		Private			*|
 \*-------------------------------------*/

/*----------------------------------------------------------------------*\
 |*			End	 					*|
 \*---------------------------------------------------------------------*/

