#include "MathTools.h"
#include "Chronos.h"
#include <cmath>
#include <iostream>
#include "produit_scalaire_tools_00.h"

using std::cout;
using std::endl;

/*----------------------------------------------------------------------*\
 |*			Declaration 					*|
 \*---------------------------------------------------------------------*/

/*--------------------------------------*\
 |*		Public			*|
 \*-------------------------------------*/
bool is_algo_produit_scalaire_ok(Algo_Pi algo, int n, string titre)
    {
    cout << "Title: " << titre << endl;
    cout << "n= " << n << endl;

    Chronos chrono;

    double produitScalaire = algo(n);
    chrono.stop();

    double produitScalaireMathematica = -24.9856;
    cout.precision(8);
    cout << "Produit scalaire= " << produitScalaire << endl;
    cout << "Produit scalaire Mathematica= " << produitScalaireMathematica << endl;

    bool isOk = MathTools::isEgale(produitScalaire, produitScalaireMathematica, 1e-6);

    cout.precision(5);
    chrono.print("time= ");

    cout << "-----------------------------------" << endl;
    return isOk;
    }

double v(long i)
    {
    return cos(sqrt(i));
    }

double w(long i)
    {
    return sin(sqrt(i));
    }

/*----------------------------------------------------------------------*\
 |*			End	 					*|
 \*---------------------------------------------------------------------*/

