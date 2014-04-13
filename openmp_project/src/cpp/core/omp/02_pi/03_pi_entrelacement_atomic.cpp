#include "pi_tools_00.h"
#include <stdio.h>
#include "omp.h"

/*----------------------------------------------------------------------*\
 |*			Declaration 					*|
 \*---------------------------------------------------------------------*/

/*--------------------------------------*\
 |*		Private			*|
 \*-------------------------------------*/
static double pi_omp_entrelacement_atomic(int n);

/*----------------------------------------------------------------------*\
 |*			Implementation 					*|
 \*---------------------------------------------------------------------*/

/*--------------------------------------*\
 |*		Public			*|
 \*-------------------------------------*/
bool pi_omp_entrelacement_atomic_ok(int n)
    {
    return is_algo_pi_ok(pi_omp_entrelacement_atomic, n, "pi_omp_entrelacement_atomic");
    }

/*--------------------------------------*\
 |*		Private			*|
 \*-------------------------------------*/
double pi_omp_entrelacement_atomic(int n)
    {

    int nbThread = omp_get_num_procs();
    omp_set_num_threads(nbThread);

    const double dx = 2 / (double) n;
    double somme = 0.0;

//Entrelacement
#pragma omp parallel
	{
	int tid = omp_get_thread_num();
	int i = tid;
	double sommeThread = 0.0;

	while (i < n)
	    {
	    double xi = -1 + i * dx;
	    sommeThread += f(xi);
	    i += nbThread;
	    }

#pragma omp atomic
	somme += sommeThread;
	}

    return 2 * dx * somme;
    }

/*----------------------------------------------------------------------*\
 |*			End	 					*|
 \*---------------------------------------------------------------------*/

