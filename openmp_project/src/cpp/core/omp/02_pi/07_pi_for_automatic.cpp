#include "pi_tools_00.h"
#include <stdio.h>
#include "omp.h"

/*----------------------------------------------------------------------*\
 |*			Declaration 					*|
 \*---------------------------------------------------------------------*/

/*--------------------------------------*\
 |*		Private			*|
 \*-------------------------------------*/
static double pi_omp_for_automatic(int n);

/*----------------------------------------------------------------------*\
 |*			Implementation 					*|
 \*---------------------------------------------------------------------*/

/*--------------------------------------*\
 |*		Public			*|
 \*-------------------------------------*/
bool pi_omp_for_automatic_ok(int n)
    {
    return is_algo_pi_ok(pi_omp_for_automatic, n, "pi_omp_for_automatic");
    }

/*--------------------------------------*\
 |*		Private			*|
 \*-------------------------------------*/
double pi_omp_for_automatic(int n)
    {

    int nbThread = omp_get_num_procs();
    omp_set_num_threads(nbThread);

    const double dx = 2 / (double) n;
    double somme = 0.0;

#pragma omp parallel for reduction(+:somme)
    for (int i = 0; i < n; i++)
	{
	double xi = -1 + i * dx;
	somme += f(xi);
	}

    return 2 * dx * somme;
    }

/*----------------------------------------------------------------------*\
 |*			End	 					*|
 \*---------------------------------------------------------------------*/
