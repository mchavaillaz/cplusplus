#ifndef SPHERE_H_
#define SPHERE_H_

#include "cudaTools.h"

class Sphere
    {
    public:

	__host__
	Sphere()
	    {
	    }

	__host__
	Sphere(float3 centre, float rayon, float hue)
	    {
	    this->centre = centre;
	    this->r = rayon;
	    this->rCarre = rayon * rayon;
	    this->hue = hue;
	    }

	__host__
	void setR(float r)
	    {
	    this->r = r;
	    this->rCarre = r * r;
	    }

	__host__
	void setCentre(float3 centre)
	    {
	    this->centre = centre;
	    }

	__host__
	void setHue(float hue)
	    {
	    this->hue = hue;
	    }

	__device__
	float hCarre(float2 xySol)
	    {
	    float a = (centre.x - xySol.x);
	    float b = (centre.y - xySol.y);
	    return a * a + b * b;
	    }

	__device__
	bool isEnDessous(float hCarre)
	    {
	    return hCarre < rCarre;
	    }

	__device__
	float dz(float hCarre)
	    {
	    return sqrtf(rCarre - hCarre);
	    }

	__device__
	float brightness(float dz)
	    {
	    return dz / r;
	    }

	__device__
	float distance(float dz)
	    {
	    return centre.z - dz;
	    }

	__device__
	float getHue()
	    {
	    return hue;
	    }

    private:
	float rCarre;
	float r;
	float3 centre;
	float hue;
    };

#endif 
