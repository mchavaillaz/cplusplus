#ifndef UTIL_H
#define UTIL_H

class Face;
class QVector3D;

class Util
{
public:
    static bool isPointInSideFace(QVector3D *pGiven, Face *face);
    static void debugLine(QVector3D *a, QVector3D *b);
    static bool sameSide(QVector3D *p, QVector3D *a, QVector3D *b, QVector3D *c);
};

#endif // UTIL_H
