#ifndef FONCTIONSMANDELBROT_H_
#define FONCTIONSMANDELBROT_H_

/*----------------------------------------------------------------------*\
 |*			Declaration 					*|
 \*---------------------------------------------------------------------*/

/*--------------------------------------*\
 |*		Private			*|
 \*-------------------------------------*/
__device__ static int calculateSequence(float x, float y, int N);
__device__ static bool isDivergent(float a, float b);
__device__ static double calculateHk(int k, int N);

/*----------------------------------------------------------------------*\
 |*			Implementation					*|
 \*---------------------------------------------------------------------*/

/*--------------------------------------*\
 |*		Private			*|
 \*-------------------------------------*/
__device__ int calculateSequence(float x, float y, int N)
    {
    float a = 0.0;
    float aCopy = 0.0;
    float b = 0.0;

    for (int i = 0; i < N; i++)
	{
	if (isDivergent(a, b))
	    {
	    return i;
	    }
	aCopy = a;
	a = (a * a - b * b) + x;
	b = 2 * aCopy * b + y;
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

