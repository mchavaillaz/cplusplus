#ifndef PRODUIT_SCALAIRE_TOOLS_00_H_
#define PRODUIT_SCALAIRE_TOOLS_00_H_
#include <iostream>

using std::string;

/*----------------------------------------------------------------------*\
 |*			Declaration 					*|
 \*---------------------------------------------------------------------*/
typedef double (*Algo_Pi)(int n);

/*--------------------------------------*\
 |*		Public			*|
 \*-------------------------------------*/
bool is_algo_produit_scalaire_ok(Algo_Pi algo, int n, string titre);
double v(long i);
double w(long i);

#endif 

/*----------------------------------------------------------------------*\
 |*			End	 					*|
 \*---------------------------------------------------------------------*/
