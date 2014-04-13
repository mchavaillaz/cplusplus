#ifndef FONCTIONSRAYTRACING_H_
#define FONCTIONSRAYTRACING_H_

/*----------------------------------------------------------------------*\
 |*			Declaration 					*|
 \*---------------------------------------------------------------------*/
#define PI_MUL_PAR_3_DIV_PAR_2 ((3.0 * M_PI)/2.0)

/*--------------------------------------*\
 |*		Private			*|
 \*-------------------------------------*/
__device__ static float hueSphere(int q, int t, float hueSphere);

/*----------------------------------------------------------------------*\
 |*			Implementation					*|
 \*---------------------------------------------------------------------*/

/*--------------------------------------*\
 |*		Private			*|
 \*-------------------------------------*/
__device__ float hueSphere(int q, int t, float hStart)
    {
    float h = 0.0;
    float T = asin(2.0 * hStart - 1.0) - PI_MUL_PAR_3_DIV_PAR_2;

    h = 0.5 + 0.5 * sin(t + PI_MUL_PAR_3_DIV_PAR_2 + T);

    return h;
    }

#endif 
