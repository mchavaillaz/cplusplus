#include "Indice2D.h"
#include "DomaineMathGPUs.h"
#include "DomaineMaths.h"
#include "IndiceXY.h"
#include "cudaTools.h"
#include "Sphere.h"
#include "ColorToolCuda.h"
#include "FonctionsRaytracing.h"

/*----------------------------------------------------------------------*\
 |*			Declaration 					*|
 \*---------------------------------------------------------------------*/

/*--------------------------------------*\
 |*		Public			*|
 \*-------------------------------------*/
__global__ void kernelFillImageRaytracing(uchar4* ptrDevImageGL, int w, int h, DomaineMathGPUs domaineMathGPUs, float t, int N, int nbSphere,
	Sphere* ptrDev_tabSphere);

/*--------------------------------------*\
 |*		Private			*|
 \*-------------------------------------*/
__device__ static void setPixel(uchar4& pixel, int i, int j, int w, int h, float t, int N, IndiceXY indiceXY, int nbSphere, Sphere* ptrDev_tabSphere);

/*----------------------------------------------------------------------*\
 |*			Implementation 					*|
 \*---------------------------------------------------------------------*/

/*--------------------------------------*\
 |*		Public			*|
 \*-------------------------------------*/
void launchKernelFillImageRaytracing(uchar4* ptrDevImageGL, int w, int h, float t, const DomaineMaths& domaineMath, int N, int nbSphere, Sphere* tabSphere, Sphere* ptrDev_tabSphere)
    {
    dim3 dg = dim3(8, 8);
    dim3 db = dim3(16, 16, 1);

    DomaineMathGPUs domaineMathGPUs(domaineMath);

kernelFillImageRaytracing<<<dg,db>>>(ptrDevImageGL, w, h, domaineMathGPUs, t, N, nbSphere, ptrDev_tabSphere);

    }

void allocMemory(Sphere* &ptrDev_tabSphere, Sphere* tabSphere, int nbSphere)
    {
    size_t sizeSphere = sizeof(Sphere) * nbSphere;

    // Device memory allocation (*)
    HANDLE_ERROR(cudaMalloc((void**) &ptrDev_tabSphere, sizeSphere));

    // Host -> Device
    HANDLE_ERROR(cudaMemcpy(ptrDev_tabSphere, tabSphere, sizeSphere, cudaMemcpyHostToDevice));
    }

void freeMemory(Sphere* ptrDev_tabSphere)
    {
    HANDLE_ERROR(cudaFree(ptrDev_tabSphere));
    }

__global__ void kernelFillImageRaytracing(uchar4* ptrDevImageGL, int w, int h, DomaineMathGPUs domaineMathGPUs, float t, int N, int nbSphere,
	Sphere* ptrDev_tabSphere)
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
	setPixel(ptrDevImageGL[s], i, j, w, h, t, N, indiceXY, nbSphere, ptrDev_tabSphere);
	ptrDevImageGL[s].w = 255;
	s += NB_THREAD;
	}
    }

/*--------------------------------------*\
 |*		Private			*|
 \*-------------------------------------*/
__device__ void setPixel(uchar4& pixelIJ, int i, int j, int w, int h, float t, int N, IndiceXY indiceXY, int nbSphere, Sphere* ptrDev_tabSphere)
    {
    float x;
    float y;
    indiceXY.toXY(i, j, x, y);

    float2 p;
    p.x = x;
    p.y = y;

    float dz;
    float b;
    float hue;
    float s;

    for (int q = 0; q < nbSphere; q++)
	{
	float hCarre = ptrDev_tabSphere[q].hCarre(p);

	if (ptrDev_tabSphere[q].isEnDessous(hCarre))
	    {
	    dz = ptrDev_tabSphere[q].dz(hCarre);
	    b = ptrDev_tabSphere[q].brightness(dz);
	    hue = hueSphere(q, t, ptrDev_tabSphere[q].getHue());
	    s = 1.0;

	    ColorToolCuda::HSB_TO_RVB(hue, s, b, pixelIJ.x, pixelIJ.y, pixelIJ.z);

	    return;
	    }
	}

    ColorToolCuda::HSB_TO_RVB(0.0, 0.0, 0.0, pixelIJ.x, pixelIJ.y, pixelIJ.z);
    }

/*----------------------------------------------------------------------*\
 |*			End	 					*|
 \*---------------------------------------------------------------------*/
