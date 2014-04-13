#include <stdio.h>
#include "omp.h"
#include "produit_scalaire_tools_00.h"

/*----------------------------------------------------------------------*\
 |*			Declaration 					*|
 \*---------------------------------------------------------------------*/

/*--------------------------------------*\
 |*		Private			*|
 \*-------------------------------------*/
static double produit_scalaire_omp_for_promotion_tab(int n);

/*----------------------------------------------------------------------*\
 |*			Implementation 					*|
 \*---------------------------------------------------------------------*/

/*--------------------------------------*\
 |*		Public			*|
 \*-------------------------------------*/
bool produit_scalaire_omp_for_promotion_tab_ok(int n)
    {
    return is_algo_produit_scalaire_ok(produit_scalaire_omp_for_promotion_tab, n, "produit_scalaire_omp_for_promotion_tab");
    }

/*--------------------------------------*\
 |*		Private			*|
 \*-------------------------------------*/
double produit_scalaire_omp_for_promotion_tab(int n)
    {
    int nbThread = omp_get_num_procs();
    omp_set_num_threads(nbThread);

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

//Syntaxe complete
#pragma omp parallel
	{
	int tid = omp_get_thread_num();
	tabResultatThread[tid] = 0;

#pragma omp for
	for (int i = 0; i < n; i++)
	    {
	    tabResultatThread[tid] += v(i) * w(i);
	    }
	}

    //Reduction de tableau
    for (int i = 0; i < nbThread; i++)
	{
	somme += tabResultatThread[i];
	}

    delete[] tabResultatThread;
    return somme;
    }

/*----------------------------------------------------------------------*\
 |*			End	 					*|
 \*---------------------------------------------------------------------*/

