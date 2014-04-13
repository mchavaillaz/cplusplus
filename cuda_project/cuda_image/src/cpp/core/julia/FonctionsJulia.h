#ifndef FONCTIONSJULIA_H_
#define FONCTIONSJULIA_H_

/*----------------------------------------------------------------------*\
 |*			Declaration 					*|
 \*---------------------------------------------------------------------*/

/*--------------------------------------*\
 |*		Private			*|
 \*-------------------------------------*/
__device__ static int calculateSequence(float x, float y, int N, float c1, float c2);
__device__ static bool isDivergent(float a, float b);
__device__ static double calculateHk(int k, int N);

/*----------------------------------------------------------------------*\
 |*			Implementation					*|
 \*---------------------------------------------------------------------*/

/*--------------------------------------*\
 |*		Private			*|
 \*-------------------------------------*/
__device__ int calculateSequence(float x, float y, int N, float c1, float c2)
    {
    float a = x;
    float aCopy = 0.0;
    float b = y;

    for (int i = 0; i < N; i++)
	{
	if (isDivergent(a, b))
	    {
	    return i;
	    }
	aCopy = a;
	a = (a * a - b * b) + c1;
	b = 2 * aCopy * b + c2;
	}

    return 0;
    }

__device__ bool isDivergent(float a, float b)
    {
    return (a * a + b * b) > 4;
    }

__device__ double calculateHk(int k, int N)
    {
    return (k / (double) N);
    }

#endif

