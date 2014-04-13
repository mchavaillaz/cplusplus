#include <stdio.h>
#include "AleaTools.h"
#include "omp.h"
#include "monte_carlo_entrelacement_atomic.h"
#include "monte_carlo_tools_00.h"

/*----------------------------------------------------------------------*\
 |*			Declaration 					*|
 \*---------------------------------------------------------------------*/


/*--------------------------------------*\
 |*		Private			*|
 \*-------------------------------------*/
static double monte_carlo_omp_entrelacement_atomic(int n);

/*----------------------------------------------------------------------*\
 |*			Implementation 					*|
 \*---------------------------------------------------------------------*/

/*--------------------------------------*\
 |*		Public			*|
 \*-------------------------------------*/
bool monte_carlo_omp_entrelacement_atomic_ok(int n)
    {
    return is_algo_monte_carlo_ok(monte_carlo_omp_entrelacement_atomic, n, "monte_carlo_omp_entrelacement_atomic");
    }

/*--------------------------------------*\
 |*		Private			*|
 \*-------------------------------------*/
double monte_carlo_omp_entrelacement_atomic(int n)
    {
    int nbThread = omp_get_num_procs();
    omp_set_num_threads(nbThread);

    double somme = 0.0;
    double sommeEInTot = 0.0;
    double sommeEOutTot = 0.0;
    double a = 0.0;
    double b = 1.0;
    double m = 8.5;
    AleaTools alea;

//Entrelacement
#pragma omp parallel
	{
	int tid = omp_get_thread_num();
	int i = tid;
	int sommeEIn = 0;
	int sommeEOut = 0;

	while (i <= n)
	    {
	    double randX = alea.uniformeAB(a, b);
	    double randY = alea.uniformeAB(0.0, m);

	    if (randY <= f(randX))
		{
		sommeEIn++;
		}
	    else
		{
		sommeEOut++;
		}

	    i += nbThread;
	    }

#pragma omp atomic
	sommeEInTot += sommeEIn;

#pragma omp atomic
	sommeEOutTot += sommeEOut;
	}

    return m * (b - a) * (sommeEInTot / (sommeEInTot + sommeEOutTot));
    }

/*----------------------------------------------------------------------*\
 |*			End	 					*|
 \*---------------------------------------------------------------------*/

