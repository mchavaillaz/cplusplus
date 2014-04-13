#ifndef FONCTIONSPRODUITSCALAIRE_H_
#define FONCTIONSPRODUITSCALAIRE_H_

/*----------------------------------------------------------------------*\
 |*			Declaration 					*|
 \*---------------------------------------------------------------------*/
#define NB_ITERATION 10000

/*--------------------------------------*\
 |*		Private			*|
 \*-------------------------------------*/
__device__ static float v(int i);
__device__ static float w(int i);
__host__ static float vHost(int i);
__host__ static float wHost(int i);

/*----------------------------------------------------------------------*\
 |*			Implementation					*|
 \*---------------------------------------------------------------------*/

/*--------------------------------------*\
 |*		Private			*|
 \*-------------------------------------*/
__device__ float v(int i)
    {
    float x = sqrtf(i);
    for(int j=1;j<=NB_ITERATION;j++)
	{
	x = cos(x);
	}
    return x;

//    return cos(sqrtf(i));
    }

__device__ float w(int i)
    {
    float x = sqrtf(i);
    for(int j=1;j<=NB_ITERATION;j++)
	{
	x = sin(x);
	}
    return x;

//    return sin(sqrtf(i));
    }

__host__ float vHost(int i)
    {
    float x = sqrtf(i);
    for(int j=1;j<=NB_ITERATION;j++)
	{
	x = cos(x);
	}
    return x;
    }

__host__ float wHost(int i)
    {
    float x = sqrtf(i);
    for(int j=1;j<=NB_ITERATION;j++)
	{
	x = sin(x);
	}
    return x;
    }

#endif 

/*----------------------------------------------------------------------*\
 |*			End	 					*|
 \*---------------------------------------------------------------------*/
