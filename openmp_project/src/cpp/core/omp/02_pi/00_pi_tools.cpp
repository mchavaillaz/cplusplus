#include "MathTools.h"
#include "Chronos.h"
#include <cmath>
#include <iostream>
#include "pi_tools_00.h"

using std::cout;
using std::endl;

/*----------------------------------------------------------------------*\
 |*			Implementation 					*|
 \*---------------------------------------------------------------------*/

/*--------------------------------------*\
 |*		Public			*|
 \*-------------------------------------*/
bool is_algo_pi_ok(Algo_Pi algo, int n, string titre)
    {
    cout << "Title: " << titre << endl;
    cout << "n= " << n << endl;

    Chronos chrono;

    double piHat = algo(n);
    chrono.stop();

    cout.precision(8);
    cout << "Pi hat= " << piHat << endl;
    cout << "Pi true= " << PI << endl;

    bool isOk = MathTools::isEgale(piHat, PI, 1e-6);

    cout.precision(3);
    chrono.print("time= ");

    cout << "---------------------------------------------" << endl;
    return isOk;
    }

double f(int x)
    {
    return sqrt(1 - x * x);
    }

/*----------------------------------------------------------------------*\
 |*			End	 					*|
 \*---------------------------------------------------------------------*/

