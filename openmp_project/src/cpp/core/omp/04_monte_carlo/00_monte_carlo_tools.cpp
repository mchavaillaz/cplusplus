#include "MathTools.h"
#include "Chronos.h"
#include <cmath>
#include <iostream>
#include "monte_carlo_tools_00.h"

using std::cout;
using std::endl;

/*----------------------------------------------------------------------*\
 |*			Implementation 					*|
 \*---------------------------------------------------------------------*/

/*--------------------------------------*\
 |*		Public			*|
 \*-------------------------------------*/
bool is_algo_monte_carlo_ok(Algo_Pi algo, int n, string titre)
    {
    cout << "Title: " << titre << endl;
    cout << "n= " << n << endl;

    Chronos chrono;

    double piMonteCarlo = algo(n);
    chrono.stop();

    cout.precision(20);
    cout << "Pi Monte carlo= " << piMonteCarlo << endl;
    cout << "Pi cmath= " << M_PI << endl;

    cout.precision(5);
    chrono.print("time= ");

    cout << "-----------------------------------" << endl;
    return true;
    }

double f(double x)
    {
    return 4 * sqrt(1 - x * x);
    }

/*----------------------------------------------------------------------*\
 |*			End	 					*|
 \*---------------------------------------------------------------------*/

