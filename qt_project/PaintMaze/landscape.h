#ifndef LANDSCAPE_H
#define LANDSCAPE_H

#include <QImage>
#include "shape_a.h"

class Landscape : public Shape_A
{
public:
    Landscape();
    ~Landscape();
    void draw();

private:
    QImage image;
    QImage image1;
    unsigned int texID;
    unsigned int texID1;
    static double edge1;

    QVector3D *p1;
    QVector3D *p2;
    QVector3D *p3;
};

#endif // LANDSCAPE_H
