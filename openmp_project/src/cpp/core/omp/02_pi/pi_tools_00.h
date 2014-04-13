#ifndef PI_TOOLS_00_H_
#define PI_TOOLS_00_H_
#include <iostream>

using std::string;

/*----------------------------------------------------------------------*\
 |*			Declaration 					*|
 \*---------------------------------------------------------------------*/
//Pointeur de fonction qui s'appelle Algo_Pi
typedef double (*Algo_Pi)(int n);

/*--------------------------------------*\
 |*		Public			*|
 \*-------------------------------------*/
double f(int x);
bool is_algo_pi_ok(Algo_Pi algo, int n, string titre);

#endif 

/*----------------------------------------------------------------------*\
 |*			End	 					*|
 \*---------------------------------------------------------------------*/
