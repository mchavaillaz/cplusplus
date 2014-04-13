#include "pi_tools_00.h"
#include <stdio.h>
#include "omp.h"

/*----------------------------------------------------------------------*\
 |*			Declaration 					*|
 \*---------------------------------------------------------------------*/

/*--------------------------------------*\
 |*		Private			*|
 \*-------------------------------------*/
static double pi_omp_for_promotion_tab(int n);

/*----------------------------------------------------------------------*\
 |*			Implementation 					*|
 \*---------------------------------------------------------------------*/

/*--------------------------------------*\
 |*		Public			*|
 \*-------------------------------------*/
bool pi_omp_for_promotion_tab_ok(int n)
    {
    return is_algo_pi_ok(pi_omp_for_promotion_tab, n, "pi_omp_for_promotion_tab");
    }

/*--------------------------------------*\
 |*		Private			*|
 \*-------------------------------------*/
double pi_omp_for_promotion_tab(int n)
    {
    int nbThread = omp_get_num_procs();
    omp_set_num_threads(nbThread);

    const double dx = 2 / (double) n;
    double somme = 0.0;
    double* tabResultatThread = new double[nbThread];

//    for (int i = 0; i < n; i++)
//	{
//	tabResultatThread[i] = 0;
//	}

////Syntaxe light
//#pragma omp parallel for
//	{
//	for (int i = 0; i < n; i++)
//	    {
//	    int tid = omp_get_thread_num();
//	    double xi = -1 + i * dx;
//
//	    tabResultatThread[tid] += f(xi);
//	    }
//
//	}

//Reduction de tableau
//    for (int i = 0; i < nbThread; i++)
//	{
//	somme += tabResultatThread[i];
//	}
//
//    delete[] tabResultatThread;
//    return 2 * dx * somme;
//    }

//Syntaxe complete
#pragma omp parallel
	{
	int tid = omp_get_thread_num();
	double xi;
	tabResultatThread[tid] = 0;
#pragma omp for
	for (int i = 0; i < n; i++)
	    {
	    xi = -1 + i * dx;
	    tabResultatThread[tid] += f(xi);
	    }
	}

    //Reduction de tableau
    for (int i = 0; i < nbThread; i++)
	{
	somme += tabResultatThread[i];
	}

    delete[] tabResultatThread;
    return 2 * dx * somme;
    }

/*----------------------------------------------------------------------*\
 |*			End	 					*|
 \*---------------------------------------------------------------------*/

