#include <stdlib.h>
#include <iostream>
#include <string>

#include "cppTest+.h"
#include "Device.h"
#include "StringTools.h"
#include "cudaTools.h"

#include "TestHelloJunit.h"


using std::string;
using std::cout;
using std::endl;

/*----------------------------------------------------------------------*\
 |*			Declaration 					*|
 \*---------------------------------------------------------------------*/

/*--------------------------------------*\
 |*		Private			*|
 \*-------------------------------------*/

static bool testALL(int deviceId);

/*--------------------------------------*\
 |*		Public			*|
 \*-------------------------------------*/

int mainTest(int deviceId);

/*----------------------------------------------------------------------*\
 |*			Implementation 					*|
 \*---------------------------------------------------------------------*/

/*--------------------------------------*\
 |*		Public			*|
 \*-------------------------------------*/

int mainTest(int deviceId)
    {
    Device::print(deviceId, "Execute on device : ");

    bool isOk = testALL(deviceId);

    cout << "\nisOK = " << isOk << endl;

    return isOk ? EXIT_SUCCESS : EXIT_FAILURE;
    }

/*--------------------------------------*\
 |*		Private			*|
 \*-------------------------------------*/

bool testALL(int deviceId)
    {
    Test::Suite testSuite;

    testSuite.add(std::auto_ptr < Test::Suite > (new TestHelloJunit(deviceId)));

    string titre = "deviceId_" + StringTools::toString(deviceId);

    return runTestHtml(titre, testSuite); // Attention: html create in working directory!!
    //return runTestConsole(titre, testSuite);
    }

/*----------------------------------------------------------------------*\
 |*			End	 					*|
 \*---------------------------------------------------------------------*/

