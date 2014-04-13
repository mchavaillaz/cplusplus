#include <cmath>
#include <QDebug>
#include "angle3d.h"
#include "movable.h"

Movable::Movable(QVector3D *p, Angle3D *angle, double vitesse) : Collidable_A(p)
{
    this->angle = angle;
    this->vitesse = vitesse;
    this->pOrigine = new QVector3D(*p);
    this->gravite = 0.01;
    this->nbTicks = 0;
}

Movable::~Movable()
{
    try
    {
    delete angle;
    }
    catch(...)
    {
        qDebug() << "destruction de movable erreur";
    }
}

void Movable::moveTick()
{
    double yaw = -angle->getYaw() * M_PI / 180.0;
    double pitch = -angle->getPitch() * M_PI / 180.0;

    double dx = sin(yaw) * vitesse;
    double dz = cos(yaw) * vitesse;
    double dy = tan(pitch) * vitesse;

    delete this->pOrigine;
    this->pOrigine = new QVector3D(*p);

    this->p->setX(this->p->x() + dx);
    this->p->setZ(this->p->z() + dz);
    this->p->setY(this->p->y() + (dy - nbTicks++*gravite));
}
