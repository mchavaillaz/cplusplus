#include <iostream>
#include <stdlib.h>
#include <assert.h>

#include "cudaTools.h"
#include "Device.h"
#include  "LimitsTools.h"


using std::cout;
using std::endl;



/*----------------------------------------------------------------------*\
 |*			Declaration 					*|
 \*---------------------------------------------------------------------*/

/*--------------------------------------*\
 |*		Imported	 	*|
 \*-------------------------------------*/

extern int mainCore(int deviceId);
extern int mainTest(int deviceId);

/*----------------------------------------------------------------------*\
 |*			Implementation 					*|
 \*---------------------------------------------------------------------*/

/*--------------------------------------*\
 |*		Private			*|
 \*-------------------------------------*/

static bool work(int deviceId, bool isTest);
static bool workALL(int nbDevice, bool isTest);


/*--------------------------------------*\
 |*		Public			*|
 \*-------------------------------------*/

int main(void)
    {
    LimitsTools::rappelTypeSize();

    if (Device::isCuda())
	{
	// Goal : Preload driver, usefull to practice  benchmarking!
	// Note : Not necessary for exe with image
	//Device::loadCudaDriverDeviceAll();// Force driver to be load for all GPU (very slow under linux)
	Device::loadCudaDriver(0); // Force driver to be load for device 0, only

	int nbDevice = Device::getDeviceCount();
	Device::printALL("All device available : ");

	bool isTest = false;
	bool isAllGPU = false;

	if (isAllGPU)
	    {
	    return workALL(nbDevice, isTest);
	    }
	else
	    {
	    int deviceId = 0;
	    assert(deviceId >= 0 && deviceId < nbDevice);

	    return work(deviceId, isTest);
	    }
	}
    else
	{
	return EXIT_FAILURE;
	}
    }

bool workALL(int nbDevice, bool isTest)
    {
    bool isOk = true;

    for (int deviceId = 0; deviceId < nbDevice; deviceId++)
	{
	isOk &= work(deviceId, isTest);
	}

    return isOk;
    }

bool work(int deviceId, bool isTest)
    {
    //HANDLE_ERROR(cudaDeviceReset());

    HANDLE_ERROR(cudaSetDevice(deviceId)); // active gpu of deviceId

    return isTest ? mainTest(deviceId) : mainCore(deviceId);
    }

/*----------------------------------------------------------------------*\
 |*			End	 					*|
 \*---------------------------------------------------------------------*/

