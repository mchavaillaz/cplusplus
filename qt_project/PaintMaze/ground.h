#ifndef GROUND_H
#define GROUND_H

#include <QImage>
#include "shape_a.h"

#include "collidable_a.h"

class Ball;

class Ground : public Collidable_A
{
public:
    Ground();
    ~Ground();
    void draw();
    bool isColliding(Collidable_A *);
    void createStain(QVector3D *p, QVector3D*, Ball *ball, double angle = 0);

private:
    QImage image;
    unsigned int texID;
    static double edge;

    QVector3D *p1;
    QVector3D *p2;
    QVector3D *p3;
};

#endif // GROUND_H
