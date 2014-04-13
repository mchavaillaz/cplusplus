#ifndef RAYTRACINGCUDAIMAGEMOOS_H_
#define RAYTRACINGCUDAIMAGEMOOS_H_

#include "ImageCudaFonctionelMOOs_A.h"
#include "DomaineMaths.h"
#include "Sphere.h"

class RaytracingCudaImageMOOs: public ImageCudaFonctionelMOOs_A
    {
    public:
	RaytracingCudaImageMOOs(unsigned int w, unsigned int h, DomaineMaths domainMath, float tStart, float dt, int N, int nbSphere);

	void fillImageGL(uchar4* ptrDevImageGL, int w, int h, const DomaineMaths& domaineMath);
	void animationStep(bool& isNeedUpdate);

	virtual ~RaytracingCudaImageMOOs();

    private:
	float t;
	float dt;
	int N;
	int nbSphere;
	Sphere* tabSphere;
	Sphere* ptrDev_tabSphere;
    };

#endif 
