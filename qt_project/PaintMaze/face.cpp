#include <cmath>
#include <QGLWidget>
#include <QDebug>
#include "face.h"
#include "util.h"

Face::Face()
{
    this->normal = 0;
    this->d = 0;
}

void Face::add(QVector3D *p)
{
    list.push_back(p);
}

void Face::projectPoint(QVector3D* OP)
{
    QVector3D *OA = this->list[2]; // pt sur face
    QVector3D *AP = new QVector3D(*OP - *OA); // xxx

    double inormal = this->normal->length();

    double normeCarre = inormal * inormal;

    // projection orthogonale d'un point sur un plan
    double num = QVector3D::dotProduct((*this->normal), (*AP));
    double denom = normeCarre;

    QVector3D x(*this->normal * (num/denom));
    QVector3D *OQ = new QVector3D(*OP - x);

    *OP = *OQ;

    OP->setX(OP->x());
    OP->setY(OP->y());
    OP->setZ(OP->z());

    delete AP;
    delete OQ;
}

bool Face::calculDistance(QVector3D *p, double *distance)
{
    QVector3D *v1 = new QVector3D((*list[0] - *list[1]));
    QVector3D *v2 = new QVector3D((*list[0] - *list.last()));

    this->normal = new QVector3D(QVector3D::crossProduct(*v2, *v1));

    delete v1;
    delete v2;

    this->d = -(normal->x()*list[0]->x() +(normal->y()*list[0]->y()) +(normal->z()*list[0]->z()));
    double num = fabs(normal->x() * p->x() + normal->y() * p->y() + normal->z() * p->z() + d);
    double denom = sqrt(p->x()*p->x() + p->y()*p->y() + p->z()*p->z());

//    delete normal;

    if(denom == 0)
        return false;

    *distance = num/denom;
    return true;

}

Face::~Face()
{
    try
    {
        qDeleteAll(list.begin(), list.end());
        delete this->normal;
    }
    catch(...)
    {
        qDebug() << "destrution de Face -> erreur";
    }
}

QVector3D * Face::getNormal()
{
    return this->normal;
}

QVector<QVector3D*>* Face::getList()
{
    return &this->list;
}
