#ifndef HISTOGRAMME_TOOLS_H_
#define HISTOGRAMME_TOOLS_H_
#include <iostream>

using std::string;

/*----------------------------------------------------------------------*\
 |*			Declaration 					*|
 \*---------------------------------------------------------------------*/
typedef unsigned int* (*Algo_Pi)(int n);

/*--------------------------------------*\
 |*		Public			*|
 \*-------------------------------------*/
void algo_histogramme(Algo_Pi algo, int n, string titre);
unsigned int* initTabData(int n);
unsigned int* getTabEmpty(int size);

#endif 

/*----------------------------------------------------------------------*\
 |*			End	 					*|
 \*---------------------------------------------------------------------*/
