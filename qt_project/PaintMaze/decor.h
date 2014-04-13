#ifndef DECOR_H
#define DECOR_H

#include <QImage>
#include "shape_a.h"
#include "point3d.h"
#include "angle3d.h"

class Decor : public Shape_A
{
public:
    Decor(QImage &image, Point3D &p1, Point3D &p2, Angle3D &angle);
    void draw();

private:
    unsigned int texID;
    Point3D p1;
    Point3D p2;
    Angle3D angle;
};

#endif // DECOR_H
