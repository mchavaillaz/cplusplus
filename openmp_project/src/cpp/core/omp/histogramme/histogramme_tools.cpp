#include "MathTools.h"
#include "AleaTools.h"
#include "Chronos.h"
#include <cmath>
#include <iostream>
#include "histogramme_tools.h"

using std::cout;
using std::endl;

/*----------------------------------------------------------------------*\
 |*			Implementation 					*|
 \*---------------------------------------------------------------------*/

/*--------------------------------------*\
 |*		Public			*|
 \*-------------------------------------*/
void algo_histogramme(Algo_Pi algo, int n, string titre)
    {
    cout << "Title: " << titre << endl;
    cout << "n= " << n << endl;

    Chronos chrono;

    unsigned int* tabFrequence = algo(n);

    for (int i = 0; i < 256; i++)
	{
	cout << "tabFrequence[" << i << "]=" << tabFrequence[i] << endl;
	}

    chrono.stop();
    chrono.print("time= ");

    cout << "-----------------------------------" << endl;
    }

unsigned int* initTabData(int n)
    {
    unsigned int* tabData = new unsigned int[n];
    AleaTools alea;

    for (int i = 0; i < n; i++)
	{
	tabData[i] = alea.uniformeAB(0, 255);
	}

    return tabData;
    }

unsigned int* getTabEmpty(int size)
    {
    unsigned int* tabFrequence = new unsigned int[size];

    for(int i=0;i<size;i++)
	{
	tabFrequence[i] = 0;
	}

    return tabFrequence;
    }

/*----------------------------------------------------------------------*\
 |*			End	 					*|
 \*---------------------------------------------------------------------*/

