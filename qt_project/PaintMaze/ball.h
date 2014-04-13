#ifndef BALL_H
#define BALL_H

#include "movable.h"

class Ball : public Movable
{
public:
    Ball(QVector3D *, Angle3D *, QColor color, double vitesse=1.5);

    virtual bool isColliding(Collidable_A *);
    virtual void draw();
    virtual void createStain(QVector3D *p, QVector3D *a, Ball *ball, double angle = 0);

    QColor getColor() { return this->color; }

private:
    QColor color;
};

#endif // BALL_H
