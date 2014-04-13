#include <stdio.h>
#include "omp.h"
#include "produit_scalaire_tools_00.h"

/*----------------------------------------------------------------------*\
 |*			Declaration 					*|
 \*---------------------------------------------------------------------*/

/*--------------------------------------*\
 |*		Public			*|
 \*-------------------------------------*/
double produit_scalaire_omp_entrelacement_promotion_tab(int n);

/*----------------------------------------------------------------------*\
 |*			Implementation					*|
 \*---------------------------------------------------------------------*/

/*--------------------------------------*\
 |*		Public			*|
 \*-------------------------------------*/
double produit_scalaire_omp_entrelacement_promotion_tab(int n)
    {

    int nbThread = omp_get_num_procs();
    omp_set_num_threads(nbThread);

    //Promotion de tableau
    double* tabResultatParThread = new double[nbThread];

//Entrelacement
#pragma omp parallel
	{
	int tid = omp_get_thread_num();
	int i = tid;
	tabResultatParThread[tid] = 0;

	while (i <= n)
	    {
	    tabResultatParThread[tid] += v(i) * w(i);
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
    return somme;
    }

bool produit_scalaire_omp_entrelacement_promotion_tab_ok(int n)
    {
    return is_algo_produit_scalaire_ok(produit_scalaire_omp_entrelacement_promotion_tab, n, "produit_scalaire_omp_entrelacement_promotion_tab");
    }

/*----------------------------------------------------------------------*\
 |*			End	 					*|
 \*---------------------------------------------------------------------*/

