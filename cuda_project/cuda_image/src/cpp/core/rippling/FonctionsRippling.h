#ifndef FONCTIONSRIPPLING_H_
#define FONCTIONSRIPPLING_H_

/*----------------------------------------------------------------------*\
 |*			Declaration 					*|
 \*---------------------------------------------------------------------*/

/*--------------------------------------*\
 |*		Private			*|
 \*-------------------------------------*/
__device__ static unsigned char levelGrey(int i, int j, int w, int h, float t);
__device__ static double d(int x, int y, int w, int h);
__device__ static double fx(int x, int w);
__device__ static double fy(int y, int h);

/*----------------------------------------------------------------------*\
 |*			Implementation					*|
 \*---------------------------------------------------------------------*/

/*--------------------------------------*\
 |*		Private			*|
 \*-------------------------------------*/
__device__ unsigned char levelGrey(int i, int j, int w, int h, float t)
    {
    double numerateur = cos((d(i, j, w, h) / 10.0) - (t / 7.0));
    double denominateur = (d(i, j, w, h) / 10.0) + 1.0;

    unsigned char color = 128 + 127 * (numerateur / denominateur);

    return color;
    }

__device__ double d(int x, int y, int w, int h)
    {
    double fX = fx(x, w);
    double fY = fy(y, h);
    return sqrt(fX * fX + fY * fY);
    }

__device__ double fx(int x, int w)
    {
    return abs(x - (w / 2.0));
    }

__device__ double fy(int y, int h)
    {
    return abs(y - (h / 2.0));
    }

#endif
