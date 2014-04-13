#include "HeatTransfert.h"
#include "cuda.h"
#include "cudaTools.h"

/*----------------------------------------------------------------------*\
 |*			Declaration 					*|
 \*---------------------------------------------------------------------*/

/*--------------------------------------*\
 |*		Public			*|
 \*-------------------------------------*/
extern void launchKernelFillImageHeatTransfert(int w, int h, float* ptrDevImageA, float* ptrDevImageB, float* ptrDevImageHeaters, float* ptrDevImageInit);
extern void launchKernelInitImageHeatTransfert(int w, int h, float* ptrDevImageA, float* ptrDevImageB, float* ptrDevImageHeaters, float* ptrDevImageInit);
extern void launchKernelRenderImageHeatTransfert(uchar4* ptrDevImageGL, int w, int h, CalibreurCudas calibreur, float* ptrDevImageToRender);

/*--------------------------------------*\
 |*		Private			*|
 \*-------------------------------------*/

/*----------------------------------------------------------------------*\
 |*			Implementation 					*|
 \*---------------------------------------------------------------------*/

/*--------------------------------------*\
 |*		Public			*|
 \*-------------------------------------*/
HeatTransfert::HeatTransfert(unsigned int w, unsigned int h, float tStart, float dt, int nbIterationAveugle) :
	ImageCudaMOOs_A(w, h), calibreur(0.0, 1.0, 0.7, 0.0)
    {
    //Input
    this->tStart = tStart;
    this->dt = dt;

    //Tools
    this->t = tStart;
    this->n = w * h;
    this->size = n * sizeof(float);
    this->nbIterationAveugle = nbIterationAveugle;
    this->cptIteration = 0;

    //Allocation Host
    ptrTabImageHeaters = new float[this->n];
    ptrTabImageA = new float[this->n];
    ptrTabImageB = new float[this->n];
    ptrTabImageInit = new float[this->n];

    //Allocation Device
    HANDLE_ERROR(cudaMalloc((void**) &ptrDevImageHeaters, this->size));
    HANDLE_ERROR(cudaMalloc((void**) &ptrDevImageA, this->size));
    HANDLE_ERROR(cudaMalloc((void**) &ptrDevImageB, this->size));
    HANDLE_ERROR(cudaMalloc((void**) &ptrDevImageInit, this->size));

    fillImageHost();

    // Host -> Device
    HANDLE_ERROR(cudaMemcpy(ptrDevImageA, ptrTabImageA, this->size, cudaMemcpyHostToDevice));
    HANDLE_ERROR(cudaMemcpy(ptrDevImageB, ptrTabImageB, this->size, cudaMemcpyHostToDevice));
    HANDLE_ERROR(cudaMemcpy(ptrDevImageHeaters, ptrTabImageHeaters, this->size, cudaMemcpyHostToDevice));
    HANDLE_ERROR(cudaMemcpy(ptrDevImageInit, ptrTabImageInit, this->size, cudaMemcpyHostToDevice));

    launchKernelInitImageHeatTransfert(w, h, this->ptrDevImageA, this->ptrDevImageB, this->ptrDevImageHeaters, this->ptrDevImageInit);
    }

HeatTransfert::~HeatTransfert()
    {
    //Desallocation Host
    delete[] ptrTabImageHeaters;
    delete[] ptrTabImageA;
    delete[] ptrTabImageB;
    delete[] ptrTabImageInit;

    ptrTabImageHeaters = NULL;
    ptrTabImageA = NULL;
    ptrTabImageB = NULL;
    ptrTabImageInit = NULL;

    //Desallocation Device
    HANDLE_ERROR(cudaFree(ptrDevImageHeaters));
    HANDLE_ERROR(cudaFree(ptrDevImageA));
    HANDLE_ERROR(cudaFree(ptrDevImageB));
    HANDLE_ERROR(cudaFree(ptrDevImageInit));
    }

void HeatTransfert::fillImageGL(uchar4 *ptrDevImageGL, int w, int h)
    {
    if (this->cptIteration < nbIterationAveugle)
	{
	this->cptIteration += 1;
	launchKernelFillImageHeatTransfert(w, h, this->ptrDevImageA, this->ptrDevImageB, this->ptrDevImageHeaters, this->ptrDevImageInit);
	}
    else
	{
	this->cptIteration = 0;
	launchKernelRenderImageHeatTransfert(ptrDevImageGL, w, h, this->calibreur, this->ptrDevImageA);
	}
    }

void HeatTransfert::animationStep(bool &isNeedUpdateView)
    {
    t += dt;
    isNeedUpdateView = true;
    }

void HeatTransfert::fillImageHost()
    {
    fillImageHeaters();
    fillImageWithValue(this->ptrTabImageInit, 0.0);
    fillImageWithValue(this->ptrTabImageA, 0.0);
    fillImageWithValue(this->ptrTabImageB, 0.0);
    }

void HeatTransfert::fillImageHeaters()
    {
    for (int i = 0; i < this->n; i++)
	{
	this->ptrTabImageHeaters[i] = 0.0;
	}

    int imageWidth = this->getW();
    int widthHeater = 50;
    int heightHeater = 50;
    int startPosition = 16 * imageWidth + 179;
    float heaterTemperature = 0.2;

    //Cold Heater
    addHeater(heightHeater, widthHeater, startPosition, heaterTemperature);

    widthHeater = 200;
    heightHeater = 200;
    startPosition = 500 * imageWidth + 500;
    heaterTemperature = 1.0;

    //Hot Heater
    addHeater(heightHeater, widthHeater, startPosition, heaterTemperature);
    }

void HeatTransfert::addHeater(int heightHeater, int widthHeater, int startPosition, float heaterTemperature)
    {
    for (int i = 0; i < heightHeater; i++)
	{
	for (int j = startPosition; j < startPosition + widthHeater; j++)
	    {
	    this->ptrTabImageHeaters[j] = heaterTemperature;
	    }
	startPosition += this->getW();
	}
    }

void HeatTransfert::fillImageWithValue(float* ptrTabImage, float value)
    {
    for (int i = 0; i < this->n; i++)
	{
	ptrTabImage[i] = value;
	}
    }

/*--------------------------------------*\
 |*		Private			*|
 \*-------------------------------------*/

/*----------------------------------------------------------------------*\
 |*			End	 					*|
 \*---------------------------------------------------------------------*/

