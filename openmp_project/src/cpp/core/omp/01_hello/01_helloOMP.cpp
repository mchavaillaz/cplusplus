#include <iostream>
#include <stdio.h>
#include "omp.h"

using std::cout;
using std::endl;

/*----------------------------------------------------------------------*\
 |*			Implementation 					*|
 \*---------------------------------------------------------------------*/

/*--------------------------------------*\
 |*		Public			*|
 \*-------------------------------------*/

void helloOMP1(void)
    {
    cout << endl << "[HelloOMP 1]" << endl;

    // Facultatif
	{
	int nbThread = omp_get_num_procs();
	omp_set_num_threads(nbThread);
	}

#pragma omp parallel
	{
	int tid = omp_get_thread_num();
	//cout pose des problèmes d'affichage... Utiliser printf plutôt
	//cout << "tid=" << tid << endl; // ligne couper
	printf("tid=%i\n", tid); //ligne non couper
	}
    }

void helloOMP2(void)
    {
    cout << endl << "[HelloOMP 2]" << endl;

    // Facultatif
	{
	int nbThread = omp_get_num_procs();
	omp_set_num_threads(nbThread);
	}

    int n = 20;
#pragma omp parallel for
    for (int i = 1; i <= n; i++)
	{
	//cout << "HelloOMP(" << i << ")" << endl; // ligne couper
	printf("HelloOMP(%i)\n", i);	// ligne non couper
	}
    }

/*----------------------------------------------------------------------*\
 |*			End	 					*|
 \*---------------------------------------------------------------------*/

