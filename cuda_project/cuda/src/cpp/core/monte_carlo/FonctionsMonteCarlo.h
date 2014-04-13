#ifndef FONCTIONSMONTECARLO_H_
#define FONCTIONSMONTECARLO_H_

/*----------------------------------------------------------------------*\
 |*			Declaration 					*|
 \*---------------------------------------------------------------------*/

/*--------------------------------------*\
 |*		Private			*|
 \*-------------------------------------*/
__device__ static float f(float x);
__device__ static float convertNumberInRange(float a, float b, float x);
__host__ static float fHost(float x);

/*----------------------------------------------------------------------*\
 |*			Implementation					*|
 \*---------------------------------------------------------------------*/

/*--------------------------------------*\
 |*		Private			*|
 \*-------------------------------------*/
__device__ float f(float x)
    {
    return 4 * sqrtf(1 - x * x);
    }

__device__ static float convertNumberInRange(float a, float b, float x)
    {
    return (((b-a))*x)+a;
    }

__host__ static float fHost(float x)
    {
    return 4 * sqrtf(1 - x * x);
    }

#endif 

/*----------------------------------------------------------------------*\
 |*			End	 					*|
 \*---------------------------------------------------------------------*/
