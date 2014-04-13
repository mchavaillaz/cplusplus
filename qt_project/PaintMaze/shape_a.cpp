#include <QDebug>
#include "shape_a.h"

Shape_A::Shape_A(QVector3D *_p)
{
    this->p = new QVector3D(_p->x(), _p->y(), _p->z());
}

Shape_A::Shape_A(const Shape_A &copy)
{
    this->p = new QVector3D(copy.getP()->x(), copy.getP()->y(), copy.getP()->z());
}

Shape_A& Shape_A::operator=(const Shape_A &copy)
{
    if(this != &copy)
    {
        if(this->p != 0)
            {
            delete this->p;
            }
        else
            {
            this->p = new QVector3D(copy.getP()->x(), copy.getP()->y(), copy.getP()->z());
            }
    }
    return *this;
}

void Shape_A::setP(QVector3D *_p)
{
    this->p->setX(_p->x());
    this->p->setY(_p->y());
    this->p->setZ(_p->z());
}

QVector3D* Shape_A::getP() const
{
    return this->p;
}

Shape_A::~Shape_A()
{
    try
    {
        delete this->p;
    }
    catch(...)
    {
        qDebug() << "destruction de shape_a erreur";
    }
}
