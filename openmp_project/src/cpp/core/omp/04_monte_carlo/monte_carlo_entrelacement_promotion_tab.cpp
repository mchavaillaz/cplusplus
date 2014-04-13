#include <stdio.h>
#include "omp.h"
#include "AleaTools.h"
#include "monte_carlo_entrelacement_promotion_tab.h"
#include "monte_carlo_tools_00.h"

/*----------------------------------------------------------------------*\
 |*			Declaration 					*|
 \*---------------------------------------------------------------------*/

/*--------------------------------------*\
 |*		Public			*|
 \*-------------------------------------*/
double monte_carlo_omp_entrelacement_promotion_tab(int n);

/*----------------------------------------------------------------------*\
 |*			Implementation					*|
 \*---------------------------------------------------------------------*/

/*--------------------------------------*\
 |*		Public			*|
 \*-------------------------------------*/
double monte_carlo_omp_entrelacement_promotion_tab(int n)
    {

    int nbThread = omp_get_num_procs();
    omp_set_num_threads(nbThread);

    //Promotion de tableau
    int* tabResultatParThreadIn = new int[nbThread];
    int* tabResultatParThreadOut = new int[nbThread];

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

	while (i <= n)
	    {
	    double randX = alea.uniformeAB(a, b);
	    double randY = alea.uniformeAB(0.0, m);

	    if (randY <= f(randX))
		{
		tabResultatParThreadIn[tid]++;
		}
	    else
		{
		tabResultatParThreadOut[tid]++;
		}

	    i += nbThread;
	    }
	}

    //Reducation sequentielle
    int sommeTotIn = 0;
    int sommeTotOut = 0;
    for (int i = 0; i < nbThread; i++)
	{
	sommeTotIn += tabResultatParThreadIn[i];
	sommeTotOut += tabResultatParThreadOut[i];
	}

    delete[] tabResultatParThreadIn;
    delete[] tabResultatParThreadOut;

    return m * (b - a) * (sommeEInTot / (sommeEInTot + sommeEOutTot));
    }

bool monte_carlo_omp_entrelacement_promotion_tab_ok(int n)
    {
    return is_algo_monte_carlo_ok(monte_carlo_omp_entrelacement_promotion_tab, n, "monte_carlo_omp_entrelacement_promotion_tab");
    }

/*----------------------------------------------------------------------*\
 |*			End	 					*|
 \*---------------------------------------------------------------------*/

