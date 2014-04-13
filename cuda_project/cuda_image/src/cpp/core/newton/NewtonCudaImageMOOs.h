#ifndef NEWTONCUDAIMAGEMOOS_H_
#define NEWTONCUDAIMAGEMOOS_H_

#include "ImageCudaFonctionelMOOs_A.h"
#include "DomaineMaths.h"

class NewtonCudaImageMOOs: public ImageCudaFonctionelMOOs_A
    {
    public:
	NewtonCudaImageMOOs(unsigned int w, unsigned int h, unsigned int N, DomaineMaths domainMath, float tStart, float dt);

	void fillImageGL(uchar4* ptrDevImageGL, int w, int h, const DomaineMaths& domaineMath);
	void animationStep(bool& isNeedUpdate);

	virtual ~NewtonCudaImageMOOs();

    private:
	float t;
	float dt;
	int N;
    };

#endif 

/*----------------------------------------------------------------------*\
 |*			End	 					*|
 \*---------------------------------------------------------------------*/
