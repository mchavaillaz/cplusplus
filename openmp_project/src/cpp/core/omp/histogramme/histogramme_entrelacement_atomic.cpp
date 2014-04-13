#include <stdio.h>
#include "AleaTools.h"
#include "omp.h"
#include "histogramme_entrelacement_atomic.h"
#include "histogramme_tools.h"

using namespace std;

/*----------------------------------------------------------------------*\
 |*			Declaration 					*|
 \*---------------------------------------------------------------------*/

/*--------------------------------------*\
 |*		Private			*|
 \*-------------------------------------*/
static unsigned int* histogramme_omp_entrelacement_atomic(int n);

/*----------------------------------------------------------------------*\
 |*			Implementation 					*|
 \*---------------------------------------------------------------------*/

/*--------------------------------------*\
 |*		Public			*|
 \*-------------------------------------*/
void launch_histogramme_omp_entrelacement_atomic(int n)
    {
    algo_histogramme(histogramme_omp_entrelacement_atomic, n, "histogramme_omp_entrelacement_atomic");
    }

/*--------------------------------------*\
 |*		Private			*|
 \*-------------------------------------*/
unsigned int* histogramme_omp_entrelacement_atomic(int n)
    {
    int nbThread = omp_get_num_procs();
    omp_set_num_threads(nbThread);

    unsigned int* tabData = initTabData(n);
    unsigned int* tabFrequence = getTabEmpty(256);

//Entrelacement
#pragma omp parallel
	{
	int tid = omp_get_thread_num();
	int i = tid;
	int j = 0;
	unsigned int* tabSommeFrequenceParThread = getTabEmpty(256);

	while (i <= n)
	    {
	    tabSommeFrequenceParThread[tabData[i]]++;
	    i += nbThread;
	    }

#pragma omp atomic
	//Je ne vois pas comment faire la reduction de tableau sans refaire une boucle for
	tabFrequence[j] += tabSommeFrequenceParThread[j];
	}

    delete[] tabData;

    return tabFrequence;
    }

/*----------------------------------------------------------------------*\
 |*			End	 					*|
 \*---------------------------------------------------------------------*/

