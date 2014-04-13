#include <stdio.h>
#include "omp.h"
#include "produit_scalaire_tools_00.h"

/*----------------------------------------------------------------------*\
 |*			Declaration 					*|
 \*---------------------------------------------------------------------*/

/*--------------------------------------*\
 |*		Private			*|
 \*-------------------------------------*/
static double produit_scalaire_omp_for_critical(int n);

/*----------------------------------------------------------------------*\
 |*			Implementation 					*|
 \*---------------------------------------------------------------------*/

/*--------------------------------------*\
 |*		Public			*|
 \*-------------------------------------*/
bool produit_scalaire_omp_for_critical_ok(int n)
    {
    return is_algo_produit_scalaire_ok(produit_scalaire_omp_for_critical, n, "produit_scalaire_omp_for_critical");
    }

/*--------------------------------------*\
 |*		Private			*|
 \*-------------------------------------*/
double produit_scalaire_omp_for_critical(int n)
    {

    int nbThread = omp_get_num_procs();
    omp_set_num_threads(nbThread);

    double somme = 0.0;

//Entrelacement
#pragma omp parallel for
    for (int i = 0; i < n; i++)
	{
	//Ici meme la fonction f est en sequentielle
#pragma omp critical (critic)
	    {
	    somme += v(i) * w(i);
	    }
	}
    return somme;
    }

/*----------------------------------------------------------------------*\
 |*			End	 					*|
 \*---------------------------------------------------------------------*/

