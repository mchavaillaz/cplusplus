#ifndef COLLIDABLE_A_H
#define COLLIDABLE_A_H

#include "shape_a.h"

class Ball;
class Stain;

class Collidable_A : public Shape_A
{
    Q_OBJECT
public:
    Collidable_A(QVector3D *p);
    virtual ~Collidable_A() {}

    virtual bool isColliding(Collidable_A *) = 0;
    virtual void draw() = 0;
    virtual void createStain(QVector3D *p, QVector3D *, Ball *ball, double angle = 0) = 0;
signals:
    void signal_addTache(Stain *);
};

#endif // COLLIDABLE_A_H
