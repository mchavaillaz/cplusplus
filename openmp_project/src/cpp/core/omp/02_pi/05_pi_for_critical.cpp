#include "pi_tools_00.h"
#include <stdio.h>
#include "omp.h"

/*----------------------------------------------------------------------*\
 |*			Declaration 					*|
 \*---------------------------------------------------------------------*/

/*--------------------------------------*\
 |*		Private			*|
 \*-------------------------------------*/
static double pi_omp_for_critical(int n);

/*----------------------------------------------------------------------*\
 |*			Implementation 					*|
 \*---------------------------------------------------------------------*/

/*--------------------------------------*\
 |*		Public			*|
 \*-------------------------------------*/
bool pi_omp_for_critical_ok(int n)
    {
    return is_algo_pi_ok(pi_omp_for_critical, n, "pi_omp_for_critical");
    }

/*--------------------------------------*\
 |*		Private			*|
 \*-------------------------------------*/
double pi_omp_for_critical(int n)
    {

    int nbThread = omp_get_num_procs();
    omp_set_num_threads(nbThread);

    const double dx = 2 / (double) n;
    double somme = 0.0;

//Entrelacement
#pragma omp parallel for
    for (int i = 0; i < n; i++)
	{
	double xi = -1 + i * dx;

	//Ici meme la fonction f est en sequentielle
#pragma omp critical (critic)
	    {
	    somme += f(xi);
	    }

	}
    return 2 * dx * somme;
    }

/*----------------------------------------------------------------------*\
 |*			End	 					*|
 \*---------------------------------------------------------------------*/
