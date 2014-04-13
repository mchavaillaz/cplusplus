#ifndef RAYTRACINGCUDAIMAGEMOOSCONSTMEMORY_H_
#define RAYTRACINGCUDAIMAGEMOOSCONSTMEMORY_H_

#include "ImageCudaFonctionelMOOs_A.h"
#include "DomaineMaths.h"
#include "Sphere.h"

class RaytracingCudaImageMOOsConstMemory: public ImageCudaFonctionelMOOs_A
    {
    public:
	RaytracingCudaImageMOOsConstMemory(unsigned int w, unsigned int h, DomaineMaths domainMath, float tStart, float dt);

	void fillImageGL(uchar4* ptrDevImageGL, int w, int h, const DomaineMaths& domaineMath);
	void animationStep(bool& isNeedUpdate);

	virtual ~RaytracingCudaImageMOOsConstMemory();

    private:
	float t;
	float dt;
    };

#endif 
