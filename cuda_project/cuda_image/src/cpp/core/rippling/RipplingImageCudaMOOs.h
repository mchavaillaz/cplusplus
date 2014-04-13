#ifndef RIPPLINGIMAGECUDAMOOS_H_
#define RIPPLINGIMAGECUDAMOOS_H_
#include "ImageCudaMOOs_A.h"

class RipplingImageCudaMOOs: public ImageCudaMOOs_A
    {
    public:
	RipplingImageCudaMOOs(unsigned int w, unsigned int h, float tStart, float dt);
	virtual ~RipplingImageCudaMOOs();

	void fillImageGL(uchar4* ptrDevImageGL, int w, int h);
	void animationStep(bool &isNeedUpdateView);

    private:
	float t;
	float dt;

    };

#endif 
