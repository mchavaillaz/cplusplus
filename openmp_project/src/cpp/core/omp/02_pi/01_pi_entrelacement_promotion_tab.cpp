#include "pi_tools_00.h"
#include <stdio.h>
#include "omp.h"

/*----------------------------------------------------------------------*\
 |*			Declaration 					*|
 \*---------------------------------------------------------------------*/

/*--------------------------------------*\
 |*		Private			*|
 \*-------------------------------------*/
static double pi_omp_entrelacement_promotion_tab(int n);

/*----------------------------------------------------------------------*\
 |*			Implementation 					*|
 \*---------------------------------------------------------------------*/

/*--------------------------------------*\
 |*		Public			*|
 \*-------------------------------------*/
bool pi_omp_entrelacement_promotion_tab_ok(int n)
    {
    return is_algo_pi_ok(pi_omp_entrelacement_promotion_tab, n, "pi_omp_entrelacement_promotion_tab");
    }

/*--------------------------------------*\
 |*		Private			*|
 \*-------------------------------------*/
double pi_omp_entrelacement_promotion_tab(int n)
    {

    int nbThread = omp_get_num_procs();
    omp_set_num_threads(nbThread);

    const double dx = 2 / (double) n;

    //Promotion de tableau
    double* tabResultatParThread = new double[nbThread];

//Entrelacement
#pragma omp parallel
	{
	int tid = omp_get_thread_num();
	int i = tid;
	tabResultatParThread[tid] = 0;

	while (i < n)
	    {
	    double xi = -1 + i * dx;
	    tabResultatParThread[tid] += f(xi);
	    i += nbThread;
	    }
	}

    //Reducation sequentielle
    double somme = 0.0;
    for (int i = 0; i < nbThread; i++)
	{
	somme += tabResultatParThread[i];
	}

    delete[] tabResultatParThread;
    return 2 * dx * somme;
    }

/*----------------------------------------------------------------------*\
 |*			End	 					*|
 \*---------------------------------------------------------------------*/

