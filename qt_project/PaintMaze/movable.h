#ifndef MOVABLE_H
#define MOVABLE_H

#include "collidable_a.h"

class QVector3D;
class Angle3D;
class QVector3D;

class Movable : public Collidable_A
{

protected :
    Angle3D * angle;
    double vitesse;

public:
    Movable(QVector3D*, Angle3D*, double vitesse);
    virtual void moveTick();

    ~Movable();
    QVector3D* getPointOrigine()const{return this->pOrigine;}

    private:
     QVector3D* pOrigine;
     double gravite;
     double nbTicks;
};

#endif // MOVABLE_H
