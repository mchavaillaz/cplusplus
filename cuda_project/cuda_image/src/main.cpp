#include <stdlib.h>
#include <iostream>
#include "ImageCudaViewers.h"
#include "RipplingImageCudaMOOs.h"
#include "MandelbrotCudaImageMOOs.h"
#include "JuliaCudaImageMOOs.h"
#include "RaytracingCudaImageMOOs.h"
#include "NewtonCudaImageMOOs.h"
#include "RaytracingCudaImageMOOsConstMemory.h"
#include "cudaTools.h"
#include "HeatTransfert.h"

using std::cout;
using std::endl;

/*----------------------------------------------------------------------*\
 |*			Importation 					*|
 \*---------------------------------------------------------------------*/

/*----------------------------------------------------------------------*\
 |*			Implementation 					*|
 \*---------------------------------------------------------------------*/

/*--------------------------------------*\
 |*		Public			*|
 \*-------------------------------------*/

/*--------------------------------------*\
 |*		Private			*|
 \*-------------------------------------*/
static void mainCore(void);
static void useRippling(void);
static void useMandelbrot(void);
static void useJulia(void);
static void useNewton(void);
static void useRaytracing(void);
static void useHeatTransfert(void);

static RipplingImageCudaMOOs getRippling(unsigned int width, unsigned int height);
static MandelbrotCudaImageMOOs getMandelbrot(unsigned int width, unsigned int height);
static JuliaCudaImageMOOs getJulia(unsigned int width, unsigned int height);
static NewtonCudaImageMOOs getNewton(unsigned int width, unsigned int height);
static RaytracingCudaImageMOOs getRaytracing(unsigned int width, unsigned int height);
static RaytracingCudaImageMOOsConstMemory getRaytracingConstMemory(unsigned int width, unsigned int height);

/*--------------------------------------*\
 |*		Private			*|
 \*-------------------------------------*/

int main(int argc, char *argv[])
    {
    ImageCudaViewers::init(argc, argv);

    mainCore();

    return EXIT_SUCCESS;
    }

void mainCore(void)
    {
//    useRippling();
//    useMandelbrot();
//    useJulia();
//    useNewton();
//    useRaytracing();
    useHeatTransfert();

//    int deviceId = 0;
//    HANDLE_ERROR(cudaSetDevice(deviceId));
//    HANDLE_ERROR(cudaGLSetGLDevice(deviceId));
//
//    unsigned int width = 600;
//    unsigned int height = 600;
//
//    bool isAnimationEnable = false;
//    bool isSelectionEnable = true;
//
//    RipplingImageCudaMOOs rippling = getRippling(width, height);
//    MandelbrotCudaImageMOOs mandelbrot = getMandelbrot(width, height);
//    JuliaCudaImageMOOs julia = getJulia(width, height);
//    NewtonCudaImageMOOs newton = getNewton(width, height);
//    RaytracingCudaImageMOOs raytracing = getRaytracing(width, height);
//    RaytracingCudaImageMOOsConstMemory raytracingConstMemory = getRaytracingConstMemory(width, height);
//
//    ImageCudaViewers imageRippling(&rippling, isAnimationEnable, isSelectionEnable);
//    ImageCudaViewers imageMandelbrot(&mandelbrot, isAnimationEnable, isSelectionEnable, width, 0);
//    ImageCudaViewers imageJulia(&julia, isAnimationEnable, isSelectionEnable, width * 2, 0);
//    ImageCudaViewers imageNewton(&newton, isAnimationEnable, isSelectionEnable, 0, height * 1.5);
//    ImageCudaViewers imageRaytracing(&raytracing, isAnimationEnable, isSelectionEnable, width, height * 1.5);
//    ImageCudaViewers imageRaytracingConstMemory(&raytracingConstMemory, isAnimationEnable, isSelectionEnable, width, 0);

    ImageCudaViewers::runALL();
    }

RipplingImageCudaMOOs getRippling(unsigned int width, unsigned int height)
    {
    float tStart = 0.0;
    float dt = 0.1;

    RipplingImageCudaMOOs rippling(width, height, tStart, dt);

    return rippling;
    }

MandelbrotCudaImageMOOs getMandelbrot(unsigned int width, unsigned int height)
    {
    float x0 = -2.1;
    float y0 = -1.3;
    float x1 = 0.8;
    float y1 = 1.3;
    DomaineMaths domaineMath(x0, y0, x1, y1);

    float tStart = 0.0;
    float dt = 0.001;
    int N = 12;

    MandelbrotCudaImageMOOs mandelbrot(width, height, domaineMath, tStart, dt, N);

    return mandelbrot;
    }

JuliaCudaImageMOOs getJulia(unsigned int width, unsigned int height)
    {
    float x0 = -1.3;
    float y0 = -1.4;
    float x1 = 1.3;
    float y1 = 1.4;
    DomaineMaths domaineMath(x0, y0, x1, y1);

    float tStart = 0.0;
    float dt = 0.0001;
    int N = 12;

    float* c = new float[2];
    c[0] = -0.12;
    c[1] = 0.85;

    JuliaCudaImageMOOs julia(width, height, domaineMath, tStart, dt, N, c);

    return julia;
    }

NewtonCudaImageMOOs getNewton(unsigned int width, unsigned int height)
    {
    //Configuration de l'animation
    float tStart = 0;
    float dt = 0.1;

    //Configuration de DomaineMaths
    double x0 = -2.0;
    double y0 = -2.0;
    double x1 = 2.0;
    double y1 = 2.0;
    DomaineMaths domaineMath(x0, y0, x1, y1);

    //Configuration pour les fractales
    int N = 12;

    NewtonCudaImageMOOs newton(width, height, N, domaineMath, tStart, dt);

    return newton;
    }

RaytracingCudaImageMOOs getRaytracing(unsigned int width, unsigned int height)
    {
    float x0 = 0.0;
    float y0 = 0.0;
    float x1 = 512.0;
    float y1 = 512.0;
    DomaineMaths domaineMath(x0, y0, x1, y1);

    float tStart = 0.0;
    float dt = 0.1;
    int N = 12;
    int nbSphere = 50;

    RaytracingCudaImageMOOs raytracing(width, height, domaineMath, tStart, dt, N, nbSphere);

    return raytracing;
    }

RaytracingCudaImageMOOsConstMemory getRaytracingConstMemory(unsigned int width, unsigned int height)
    {
    float x0 = 0.0;
    float y0 = 0.0;
    float x1 = 512.0;
    float y1 = 512.0;
    DomaineMaths domaineMath(x0, y0, x1, y1);

    float tStart = 0.0;
    float dt = 0.1;
    int N = 12;
    int nbSphere = 50;

    RaytracingCudaImageMOOsConstMemory raytracingConstMemory(width, height, domaineMath, tStart, dt);

    return raytracingConstMemory;
    }

void useRippling(void)
    {
    int deviceId = 0;
    HANDLE_ERROR(cudaSetDevice(deviceId));
    HANDLE_ERROR(cudaGLSetGLDevice(deviceId));

    unsigned int imageWidth = 512;
    unsigned int imageHeight = 512;
    float tStart = 0.0;
    float dt = 1.0;

    RipplingImageCudaMOOs rippling(imageWidth, imageHeight, tStart, dt);

    bool isAnimationEnable = true;

    ImageCudaViewers imageCudaViewer(&rippling, isAnimationEnable);
    ImageCudaViewers::runALL();
    }

void useMandelbrot(void)
    {
    int deviceId = 0;
    HANDLE_ERROR(cudaSetDevice(deviceId));
    HANDLE_ERROR(cudaGLSetGLDevice(deviceId));

    unsigned int imageWidth = 512;
    unsigned int imageHeight = 512;

    float x0 = -2.1;
    float y0 = -1.3;
    float x1 = 0.8;
    float y1 = 1.3;
    DomaineMaths domaineMath(x0, y0, x1, y1);

    float tStart = 0.0;
    float dt = 0.0001;
    int N = 12;

    MandelbrotCudaImageMOOs mandelbrot(imageWidth, imageHeight, domaineMath, tStart, dt, N);

    bool isAnimationEnable = true;
    bool isSelectionEnable = true;

    ImageCudaViewers imageCudaViewer(&mandelbrot, isAnimationEnable, isSelectionEnable);
    ImageCudaViewers::runALL();
    }

void useJulia(void)
    {
    int deviceId = 0;
    HANDLE_ERROR(cudaSetDevice(deviceId));
    HANDLE_ERROR(cudaGLSetGLDevice(deviceId));

    unsigned int imageWidth = 512;
    unsigned int imageHeight = 512;

    float x0 = -1.3;
    float y0 = -1.4;
    float x1 = 1.3;
    float y1 = 1.4;
    DomaineMaths domaineMath(x0, y0, x1, y1);

    float tStart = 0.0;
    float dt = 0.0001;
    int N = 12;

    float* c = new float[2];
    c[0] = -0.12;
    c[1] = 0.85;

    JuliaCudaImageMOOs julia(imageWidth, imageHeight, domaineMath, tStart, dt, N, c);

    bool isAnimationEnable = true;
    bool isSelectionEnable = true;

    ImageCudaViewers imageCudaViewer(&julia, isAnimationEnable, isSelectionEnable);
    ImageCudaViewers::runALL();
    }

void useNewton(void)
    {
    int deviceId = 0;
    HANDLE_ERROR(cudaSetDevice(deviceId));
    HANDLE_ERROR(cudaGLSetGLDevice(deviceId));

    //Configuration de la taille de l'ecran
    unsigned int w = 512;
    unsigned int h = 512;

    //Configuration de l'animation
    float tStart = 0;
    float dt = 0.1;

    //Configuration de DomaineMaths
    double x0 = -2.0;
    double y0 = -2.0;
    double x1 = 2.0;
    double y1 = 2.0;
    DomaineMaths domaineMath(x0, y0, x1, y1);

    //Configuration pour les fractales
    int N = 12;

    bool isAnimationEnable = true;
    bool isSelectionEnable = true;

    NewtonCudaImageMOOs newton(w, h, N, domaineMath, tStart, dt);

    ImageCudaViewers imageCudaViewer(&newton, isAnimationEnable, isSelectionEnable);
    ImageCudaViewers::runALL();
    }

void useRaytracing(void)
    {
    int deviceId = 0;
    HANDLE_ERROR(cudaSetDevice(deviceId));
    HANDLE_ERROR(cudaGLSetGLDevice(deviceId));

    unsigned int imageWidth = 512;
    unsigned int imageHeight = 512;

    float x0 = 0.0;
    float y0 = 0.0;
    float x1 = 512.0;
    float y1 = 512.0;
    DomaineMaths domaineMath(x0, y0, x1, y1);

    float tStart = 0.0;
    float dt = 0.001;
    int N = 12;
    int nbSphere = 50;

    RaytracingCudaImageMOOs raytracing(imageWidth, imageHeight, domaineMath, tStart, dt, N, nbSphere);

    bool isAnimationEnable = true;
    bool isSelectionEnable = true;

    ImageCudaViewers imageCudaViewer(&raytracing, isAnimationEnable, isSelectionEnable);
    ImageCudaViewers::runALL();
    }

void useHeatTransfert(void)
    {
    int deviceId = 0;
    HANDLE_ERROR(cudaSetDevice(deviceId));
    HANDLE_ERROR(cudaGLSetGLDevice(deviceId));

    unsigned int imageWidth = 800;
    unsigned int imageHeight = 800;

    float tStart = 0;
    float dt = 0.1;

    HeatTransfert heatTransfert(imageWidth, imageHeight, tStart, dt, 50);

    bool isAnimationEnable = true;
    bool isSelectionEnable = true;

    ImageCudaViewers imageCudaViewer(&heatTransfert, isAnimationEnable, isSelectionEnable);
    ImageCudaViewers::runALL();
    }

/*----------------------------------------------------------------------*\
 |*			End	 					*|
 \*---------------------------------------------------------------------*/

