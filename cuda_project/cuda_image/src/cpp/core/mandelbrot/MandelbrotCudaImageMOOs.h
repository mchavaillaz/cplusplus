#ifndef MANDELBROTCUDAIMAGEMOOS_H_
#define MANDELBROTCUDAIMAGEMOOS_H_

#include "ImageCudaFonctionelMOOs_A.h"
#include "DomaineMaths.h"

class MandelbrotCudaImageMOOs: public ImageCudaFonctionelMOOs_A
    {
    public:
	MandelbrotCudaImageMOOs(unsigned int w, unsigned int h, DomaineMaths domainMath, float tStart, float dt, int N);

	void fillImageGL(uchar4* ptrDevImageGL, int w, int h, const DomaineMaths& domaineMath);
	void animationStep(bool& isNeedUpdate);

	virtual ~MandelbrotCudaImageMOOs();

    private:
	float t;
	float dt;
	int N;
    };

#endif 

/*----------------------------------------------------------------------*\
 |*			End	 					*|
 \*---------------------------------------------------------------------*/
