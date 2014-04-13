#ifndef STAIN_H
#define STAIN_H

#include <QVector>
#include <QColor>
#include "shape_a.h"

class Face;
class QVector3D;
class QGLBuffer;

class Stain : public Shape_A
{
public:
    Stain(Face *, QVector3D *, QVector3D*, QColor *);
    void draw();
    QColor getColor() {return this->color;}
    Face* getFace()const{return this->face;}

private:
    Face *face;
    QVector3D *pModif;
    QVector3D *pOrigine;
    QVector<QVector3D*> points;
    QColor color;
    int taille;

    QGLBuffer *buffer;
    QGLBuffer *buffer2;

    double getRandomRayon();
    double getRandomAlpha();
    double getRandomBeta();
};

#endif // STAIN_H
