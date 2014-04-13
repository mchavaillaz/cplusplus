#include <stdio.h>
#include "AleaTools.h"
#include "omp.h"
#include "histogramme_for_atomic.h"
#include "histogramme_tools.h"

using namespace std;

/*----------------------------------------------------------------------*\
 |*			Declaration 					*|
 \*---------------------------------------------------------------------*/

/*--------------------------------------*\
 |*		Private			*|
 \*-------------------------------------*/
static unsigned int* histogramme_omp_for_atomic(int n);

/*----------------------------------------------------------------------*\
 |*			Implementation 					*|
 \*---------------------------------------------------------------------*/

/*--------------------------------------*\
 |*		Public			*|
 \*-------------------------------------*/
void launch_histogramme_omp_for_atomic(int n)
    {
    algo_histogramme(histogramme_omp_for_atomic, n, "histogramme_omp_for_atomic");
    }

/*--------------------------------------*\
 |*		Private			*|
 \*-------------------------------------*/
unsigned int* histogramme_omp_for_atomic(int n)
    {
    int nbThread = omp_get_num_procs();
    omp_set_num_threads(nbThread);

    unsigned int* tabData = initTabData(n);
    unsigned int* tabFrequence = getTabEmpty(256);


//Syntaxe complete
#pragma omp parallel
	{

#pragma omp for
	for (long i = 0; i < n; i++)
	    {
#pragma omp atomic

	    tabFrequence[tabData[i]]++;
	    }
	}

    return tabFrequence;
    }

/*----------------------------------------------------------------------*\
 |*			End	 					*|
 \*---------------------------------------------------------------------*/

