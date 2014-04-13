#include <stdio.h>
#include "AleaTools.h"
#include "omp.h"
#include "histogramme_entrelacement_promotion_tab.h"
#include "histogramme_tools.h"

using namespace std;

/*----------------------------------------------------------------------*\
 |*			Declaration 					*|
 \*---------------------------------------------------------------------*/

/*--------------------------------------*\
 |*		Private			*|
 \*-------------------------------------*/
static unsigned int* histogramme_omp_entrelacement_promotion_tab(int n);

/*----------------------------------------------------------------------*\
 |*			Implementation 					*|
 \*---------------------------------------------------------------------*/

/*--------------------------------------*\
 |*		Public			*|
 \*-------------------------------------*/
void launch_histogramme_omp_entrelacement_promotion_tab(int n)
    {
    algo_histogramme(histogramme_omp_entrelacement_promotion_tab, n, "histogramme_omp_entrelacement_promotion_tab");
    }

/*--------------------------------------*\
 |*		Private			*|
 \*-------------------------------------*/
unsigned int* histogramme_omp_entrelacement_promotion_tab(int n)
    {
    int nbThread = omp_get_num_procs();
    omp_set_num_threads(nbThread);

    unsigned int* tabData = initTabData(n);
    unsigned int* tabFrequence = getTabEmpty(256);
    unsigned int** tabSommeFrequenceParThread = new unsigned int*[nbThread];

    for (int j = 0; j < nbThread; j++)
	{
	tabSommeFrequenceParThread[j] = getTabEmpty(256);
	}

//Entrelacement
#pragma omp parallel
	{
	int tid = omp_get_thread_num();
	int i = tid;

	while (i <= n)
	    {
	    tabSommeFrequenceParThread[tid][tabData[i]]++;

	    i += nbThread;
	    }
	}

    //Reducation sequentielle
    for (int m = 0; m < nbThread; m++)
	{
	for (int k = 0; k < 256; k++)
	    {
	    tabFrequence[k] += tabSommeFrequenceParThread[m][k];
	    }
	}

    for (int m = 0; m < nbThread; m++)
	{
	delete tabSommeFrequenceParThread[m];
	}

    delete[] tabSommeFrequenceParThread;
    delete[] tabData;

    return tabFrequence;
    }

/*----------------------------------------------------------------------*\
 |*			End	 					*|
 \*---------------------------------------------------------------------*/

