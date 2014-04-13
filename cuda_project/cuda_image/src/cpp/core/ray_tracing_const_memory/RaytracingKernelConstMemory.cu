#include "Indice2D.h"
#include "DomaineMathGPUs.h"
#include "DomaineMaths.h"
#include "IndiceXY.h"
#include "cudaTools.h"
#include "SphereConstMemory.h"
#include <ctime>
#include "ColorToolCuda.h"

/*----------------------------------------------------------------------*\
 |*			Declaration 					*|
 \*---------------------------------------------------------------------*/

/*--------------------------------------*\
 |*		Public			*|
 \*-------------------------------------*/
__global__ void kernelFillImageRaytracingConstMemory(uchar4* ptrDevImageGL, int w, int h, DomaineMathGPUs domaineMathGPUs, float t);

/*--------------------------------------*\
 |*		Constant		*|
 \*-------------------------------------*/
#define NB_SPHERE 50
__constant__ Sphere TAB_SPHERE_CST[NB_SPHERE];

/*--------------------------------------*\
 |*		Private			*|
 \*-------------------------------------*/
__device__ static void setPixel(uchar4& pixel, int i, int j, int w, int h, float t, IndiceXY indiceXY);

/*----------------------------------------------------------------------*\
 |*			Implementation 					*|
 \*---------------------------------------------------------------------*/

/*--------------------------------------*\
 |*		Public			*|
 \*-------------------------------------*/
void launchKernelFillImageRaytracingConstMemory(uchar4* ptrDevImageGL, int w, int h, float t, const DomaineMaths& domaineMath)
    {
    dim3 dg = dim3(8, 8);
    dim3 db = dim3(16, 16, 1);

    DomaineMathGPUs domaineMathGPUs(domaineMath);

kernelFillImageRaytracingConstMemory<<<dg,db>>>(ptrDevImageGL, w, h, domaineMathGPUs, t);

    }

void allocMemoryConstMemory(unsigned int w, unsigned int h)
    {
    Sphere tabSphere[NB_SPHERE];

    srand((unsigned int) time(NULL));
    float rayonAlea = 0.0;
    float hueAlea = 0.0;
    float3 p;

    //Creation des Spheres
    for (int i = 0; i < NB_SPHERE; i++)
    {
	rayonAlea = (rand() % (w / 8)) + 10.0;
	hueAlea = (rand() % 1000) / 1000.0;

	p.x = rand() % w;
	p.y = rand() % h;
	p.z = (rand() % (2 * w)) + 10.0;

	tabSphere[i].setR(rayonAlea);
	tabSphere[i].setHue(hueAlea);
	tabSphere[i].setCentre(p);
    }

    size_t sizeSphere = sizeof(Sphere) * NB_SPHERE;
    int offset = 0;

    // Host -> Device
    HANDLE_ERROR(cudaMemcpyToSymbol(TAB_SPHERE_CST, &tabSphere, sizeSphere, offset, cudaMemcpyHostToDevice));
    }

void freeMemoryConstMemory()
    {
//    HANDLE_ERROR(cudaFree(ptrDev_tabSphere));
    }

__global__ void kernelFillImageRaytracingConstMemory(uchar4* ptrDevImageGL, int w, int h, DomaineMathGPUs domaineMathGPUs, float t)
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
	setPixel(ptrDevImageGL[s], i, j, w, h, t, indiceXY);
	ptrDevImageGL[s].w = 255;
	s += NB_THREAD;
	}
    }

/*--------------------------------------*\
 |*		Private			*|
 \*-------------------------------------*/
__device__ void setPixel(uchar4& pixelIJ, int i, int j, int w, int h, float t, IndiceXY indiceXY)
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

    for (int q = 0; q < NB_SPHERE; q++)
	{
	float hCarre = TAB_SPHERE_CST[q].hCarre(p);

	if (TAB_SPHERE_CST[q].isEnDessous(hCarre))
	    {
	    dz = TAB_SPHERE_CST[q].dz(hCarre);
	    b = TAB_SPHERE_CST[q].brightness(dz);
	    hue = TAB_SPHERE_CST[q].getHue();
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
