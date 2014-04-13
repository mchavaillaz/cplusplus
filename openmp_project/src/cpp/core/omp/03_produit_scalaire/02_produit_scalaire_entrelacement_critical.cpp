#include <stdio.h>
#include "omp.h"
#include "produit_scalaire_tools_00.h"

/*----------------------------------------------------------------------*\
 |*			Declaration 					*|
 \*---------------------------------------------------------------------*/

/*--------------------------------------*\
 |*		Private			*|
 \*-------------------------------------*/
static double produit_scalaire_omp_entrelacement_critical(int n);

/*----------------------------------------------------------------------*\
 |*			Implementation 					*|
 \*---------------------------------------------------------------------*/

/*--------------------------------------*\
 |*		Public			*|
 \*-------------------------------------*/
bool produit_scalaire_omp_entrelacement_critical_ok(int n)
    {
    return is_algo_produit_scalaire_ok(produit_scalaire_omp_entrelacement_critical, n, "produit_scalaire_omp_entrelacement_critical");
    }

/*--------------------------------------*\
 |*		Private			*|
 \*-------------------------------------*/
double produit_scalaire_omp_entrelacement_critical(int n)
    {
    int nbThread = omp_get_num_procs();
    omp_set_num_threads(nbThread);

    //On dit que la variable somme est partagee car elle est accessible pour chaque thread
    double somme = 0.0;

//Entrelacement
#pragma omp parallel
	{
	//Les variables ici sont private
	int tid = omp_get_thread_num();
	int i = tid;
	double sommeThread = 0.0;

	while (i <= n)
	    {
	    sommeThread += v(i) * w(i);
	    i += nbThread;
	    }
	//Problme d'acces concurent sur somme
	//Ce bout de code gere l'acces concurent sur la variable somme car elle n'est pas dans omp parallel
#pragma omp critical
	    {
	    somme += sommeThread;
	    }
	}

    return somme;
    }

/*----------------------------------------------------------------------*\
 |*			End	 					*|
 \*---------------------------------------------------------------------*/

