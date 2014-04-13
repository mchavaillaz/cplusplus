#include <stdio.h>
#include "omp.h"
#include "produit_scalaire_tools_00.h"

/*----------------------------------------------------------------------*\
 |*			Declaration 					*|
 \*---------------------------------------------------------------------*/

/*--------------------------------------*\
 |*		Private			*|
 \*-------------------------------------*/
static double produit_scalaire_omp_entrelacement_atomic(int n);

/*----------------------------------------------------------------------*\
 |*			Implementation 					*|
 \*---------------------------------------------------------------------*/

/*--------------------------------------*\
 |*		Public			*|
 \*-------------------------------------*/
bool produit_scalaire_omp_entrelacement_atomic_ok(int n)
    {
    return is_algo_produit_scalaire_ok(produit_scalaire_omp_entrelacement_atomic, n, "produit_scalaire_omp_entrelacement_atomic");
    }

/*--------------------------------------*\
 |*		Private			*|
 \*-------------------------------------*/
double produit_scalaire_omp_entrelacement_atomic(int n)
    {
    int nbThread = omp_get_num_procs();
    omp_set_num_threads(nbThread);

    double somme = 0.0;

//Entrelacement
#pragma omp parallel
	{
	int tid = omp_get_thread_num();
	int i = tid;
	double sommeThread = 0.0;

	while (i <= n)
	    {
	    sommeThread += v(i) * w(i);
	    i += nbThread;
	    }

	//Problme d'acces concurent sur somme
	//Ici c'est l'affectation qui sera sequentielle
#pragma omp atomic
	somme += sommeThread;
	}

    return somme;
    }

/*----------------------------------------------------------------------*\
 |*			End	 					*|
 \*---------------------------------------------------------------------*/

