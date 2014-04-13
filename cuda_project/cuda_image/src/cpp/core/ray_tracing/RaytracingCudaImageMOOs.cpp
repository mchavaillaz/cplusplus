#include "RaytracingCudaImageMOOs.h"
#include <cmath>
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
extern void launchKernelFillImageRaytracing(uchar4* ptrDevImageGL, int w, int h, float t, const DomaineMaths& domaineMath, int N, int nbSphere,
	Sphere* tabSphere, Sphere* ptrDev_tabSphere);
extern void allocMemory(Sphere*& ptrDev_tabSphere, Sphere* tabSphere, int nbSphere);
extern void freeMemory(Sphere* ptrDev_tabSphere);

/*----------------------------------------------------------------------*\
 |*			Implementation 					*|
 \*---------------------------------------------------------------------*/

/*--------------------------------------*\
 |*		Public			*|
 \*-------------------------------------*/
RaytracingCudaImageMOOs::RaytracingCudaImageMOOs(unsigned int w, unsigned int h, DomaineMaths domainMath, float tStart, float dt, int N, int nbSphere) :
	ImageCudaFonctionelMOOs_A(w, h, domainMath)
    {
    this->t = tStart;
    this->dt = dt;
    this->N = N;
    this->nbSphere = nbSphere;

    tabSphere = new Sphere[this->nbSphere];

    srand((unsigned int) time(NULL));
    float rayonAlea = 0.0;
    float hueAlea = 0.0;
    float3 p;

    //Création des Spheres
    for (int i = 0; i < this->nbSphere; i++)
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

    allocMemory(this->ptrDev_tabSphere, this->tabSphere, this->nbSphere);
    }

void RaytracingCudaImageMOOs::fillImageGL(uchar4* ptrDevImageGL, int w, int h, const DomaineMaths& domaineMath)
    {
    launchKernelFillImageRaytracing(ptrDevImageGL, w, h, this->t, domaineMath, this->N, this->nbSphere, this->tabSphere, this->ptrDev_tabSphere);
    }

void RaytracingCudaImageMOOs::animationStep(bool& isNeedUpdate)
    {
    this->t += dt;
    isNeedUpdate = true;
    }

RaytracingCudaImageMOOs::~RaytracingCudaImageMOOs()
    {
    delete[] tabSphere;
    freeMemory(this->ptrDev_tabSphere);
    }

/*----------------------------------------------------------------------*\
 |*			End	 					*|
 \*---------------------------------------------------------------------*/

