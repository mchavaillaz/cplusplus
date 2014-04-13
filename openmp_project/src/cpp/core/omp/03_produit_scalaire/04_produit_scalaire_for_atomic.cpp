#include <stdio.h>
#include "omp.h"
#include "produit_scalaire_tools_00.h"

/*----------------------------------------------------------------------*\
 |*			Declaration 					*|
 \*---------------------------------------------------------------------*/

/*--------------------------------------*\
 |*		Private			*|
 \*-------------------------------------*/
static double produit_scalaire_omp_for_atomic(int n);

/*----------------------------------------------------------------------*\
 |*			Implementation 					*|
 \*---------------------------------------------------------------------*/

/*--------------------------------------*\
 |*		Public			*|
 \*-------------------------------------*/
bool produit_scalaire_omp_for_atomic_ok(int n)
    {
    return is_algo_produit_scalaire_ok(produit_scalaire_omp_for_atomic, n, "produit_scalaire_omp_for_atomic");
    }

/*--------------------------------------*\
 |*		Private			*|
 \*-------------------------------------*/
double produit_scalaire_omp_for_atomic(int n)
    {
    int nbThread = omp_get_num_procs();
    omp_set_num_threads(nbThread);

    double somme = 0.0;

//Syntaxe light
//#pragma omp parallel for
//	{
//	for (int i = 0; i < n; i++)
//	    {
//	    double xi = -1 + i * dx;
//
//	    //Gere les acces concurents sur la variables somme
//	    //somme est sequentielle
//	    //f(xi) est parallele
//	    //Fait de la synchronisation
//#pragma omp atomic
//		{
//		somme += f(xi);
//		}
//	    }
//
//	}
//
//    return 2 * dx * somme;
//    }

//Syntaxe complete
#pragma omp parallel
	{
#pragma omp for
	for (int i = 0; i < n; i++)
	    {
#pragma omp atomic
	    somme += v(i) * w(i);
	    }

	}
    return somme;
    }

/*----------------------------------------------------------------------*\
 |*			End	 					*|
 \*---------------------------------------------------------------------*/

