#include "pi_tools_00.h"
#include <stdio.h>
#include "omp.h"

/*----------------------------------------------------------------------*\
 |*			Declaration 					*|
 \*---------------------------------------------------------------------*/

/*--------------------------------------*\
 |*		Private			*|
 \*-------------------------------------*/
static double pi_omp_for_atomic(int n);

/*----------------------------------------------------------------------*\
 |*			Implementation 					*|
 \*---------------------------------------------------------------------*/

/*--------------------------------------*\
 |*		Public			*|
 \*-------------------------------------*/
bool pi_omp_for_atomic_ok(int n)
    {
    return is_algo_pi_ok(pi_omp_for_atomic, n, "pi_omp_for_atomic");
    }

/*--------------------------------------*\
 |*		Private			*|
 \*-------------------------------------*/
double pi_omp_for_atomic(int n)
    {
    int nbThread = omp_get_num_procs();
    omp_set_num_threads(nbThread);

    const double dx = 2 / (double) n;
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
	    double xi = -1 + i * dx;
#pragma omp atomic
	    somme += f(xi);
	    }

	}
    return 2 * dx * somme;
    }

/*----------------------------------------------------------------------*\
 |*			End	 					*|
 \*---------------------------------------------------------------------*/
