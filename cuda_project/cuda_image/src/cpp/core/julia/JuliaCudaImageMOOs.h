#ifndef JULIACUDAIMAGEMOOS_H_
#define JULIACUDAIMAGEMOOS_H_

#include "ImageCudaFonctionelMOOs_A.h"
#include "DomaineMaths.h"

class JuliaCudaImageMOOs: public ImageCudaFonctionelMOOs_A
    {
    public:
	JuliaCudaImageMOOs(unsigned int w, unsigned int h, DomaineMaths domainMath, float tStart, float dt, int N, float* c);

	void fillImageGL(uchar4* ptrDevImageGL, int w, int h, const DomaineMaths& domaineMath);
	void animationStep(bool& isNeedUpdate);

	virtual ~JuliaCudaImageMOOs();

    private:
	float t;
	float dt;
	int N;
	float* c;
    };

#endif 

/*----------------------------------------------------------------------*\
 |*			End	 					*|
 \*---------------------------------------------------------------------*/
