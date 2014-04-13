#ifndef FONCTIONSNEWTON_H_
#define FONCTIONSNEWTON_H_

/*----------------------------------------------------------------------*\
 |*			Declaration 					*|
 \*---------------------------------------------------------------------*/

/*--------------------------------------*\
 |*		Private			*|
 \*-------------------------------------*/
__device__ static float calculateSequence(float x, float y, int N);
__device__ static bool isDivergent(float xi, float yi, float xii, float yii);
__device__ static void calculXY(float x, float y, float &xOut, float &yOut);

/*----------------------------------------------------------------------*\
 |*			Implementation					*|
 \*---------------------------------------------------------------------*/

/*--------------------------------------*\
 |*		Private			*|
 \*-------------------------------------*/
__device__ float calculateSequence(float x, float y, int N)
    {
    float xii = x;
    float yii = y;
    float xi = x;
    float yi = y;

    float sqrt3par2 = sqrt(3.0) / 2.0;

    for (int i = 0; i < N; i++)
	{
	xi = xii;
	yi = yii;
	calculXY(xi, yi, xii, yii);
	if (isDivergent(xi, yi, xii, yii))
	    {
	    break;
	    }
	}

    float distxa = ((xii - 1.0) * (xii - 1.0)) + (yii * yii);
    float distxb = ((xii + 0.5) * (xii + 0.5)) + ((yii - sqrt3par2) * (yii - sqrt3par2));
    float distxc = ((xii + 0.5) * (xii + 0.5)) + ((yii + sqrt3par2) * (yii + sqrt3par2));

    float res = fmin(distxa, fmin(distxb, distxc));

    if (res == distxa)
	{
	return 0.0;
	}
    else if (res == distxb)
	{
	return 0.5;
	}
    else
	{
	return 1.0;
	}
    }

__device__ bool isDivergent(float xi, float yi, float xii, float yii)
    {
    double numerateur = ((xi - xii) * (xi - xii)) + ((yi - yii) * (yi - yii));
    double denominateur = xii * xii + yii * yii;

    return (numerateur / denominateur) < 1e-5;
    }

__device__ void calculXY(float x, float y, float &xOut, float &yOut)
    {
    float f1xy = (x * x * x) - (3.0 * x * y * y) - 1.0;
    float f2xy = (y * y * y) - (3.0 * x * x * y);

    float a = (3.0 * x * x) - (3.0 * y * y);
    float b = (-6.0) * x * y;
    float c = (-6.0) * x * y;
    float d = (3.0 * y * y) - (3.0 * x * x);

    float detA = (a * d) - (b * c);

    //inverse du determinant
    float invDetA = 1.0 / detA;

    //inverse
    float temp = a;
    a = d;
    d = temp;

    c = -c;
    b = -b;

    float aNew = invDetA * a;
    float bNew = invDetA * b;
    float cNew = invDetA * c;
    float dNew = invDetA * d;

    xOut = x - (aNew * f1xy + bNew * f2xy);
    yOut = y - (cNew * f1xy + dNew * f2xy);
    }

#endif

