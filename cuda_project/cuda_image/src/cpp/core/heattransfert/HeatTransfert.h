#ifndef HEATTRANSFERT_H_
#define HEATTRANSFERT_H_

#include "ImageCudaMOOs_A.h"
#include "CalibreurCudas.h"

using namespace std;
/*----------------------------------------------------------------------*\
 |*			Declaration 					*|
 \*---------------------------------------------------------------------*/

/*--------------------------------------*\
 |*		Public			*|
 \*-------------------------------------*/

class HeatTransfert: public ImageCudaMOOs_A
    {
    public:
	HeatTransfert(unsigned int w, unsigned int h, float tStart, float dt, int nbIterationAveugle);

	void fillImageGL(uchar4* ptrDevImageGL, int w, int h);
	void animationStep(bool &isNeedUpdateView);

	virtual ~HeatTransfert();

    private:

	void fillImageHost();
	void fillImageHeaters();
	void fillImageWithValue(float* ptrTabImage, float value);
	void addHeater(int heightHeater, int widthHeater, int startPosition, float heaterTemperature);

	//Input
	float tStart;
	float dt;

	//Tools
	size_t size;
	int n;
	float t;
	CalibreurCudas calibreur;
	int nbIterationAveugle;
	int cptIteration;

	float* ptrTabImageHeaters;
	float* ptrTabImageInit;
	float* ptrTabImageA;
	float* ptrTabImageB;

	float* ptrDevImageA;
	float* ptrDevImageB;
	float* ptrDevImageHeaters;
	float* ptrDevImageInit;
    };

#endif 

/*----------------------------------------------------------------------*\
 |*			End	 					*|
 \*---------------------------------------------------------------------*/
