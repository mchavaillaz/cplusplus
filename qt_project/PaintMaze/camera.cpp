#include <QGLWidget>
#include <QVector3D>
#include <QDebug>
#include "camera.h"
#include "mainglwidget.h"


Camera::Camera(QVector3D *p, Angle3D *a)
{
    this->p = new QVector3D(p->x(), p->y(), p->z());
    this->a = new Angle3D(*a);
}

void Camera::view() {
    glRotated(this->a->getPitch(), 1, 0, 0);
    glRotated(this->a->getYaw(), 0, 1, 0);
    glRotated(this->a->getRoll(), 0, 0, 1);

    glTranslated(this->p->x(), -this->p->y(), this->p->z());
}

QVector3D* Camera::getPoint()
{
    return this->p;
}

Angle3D* Camera::getAngle()
{
    return this->a;
}

void Camera::setYaw(double yaw)
{
    while(yaw < 0)
        yaw += 360;

    while(yaw >= 360)
        yaw -= 360;

    this->a->setYaw(yaw);
}

void Camera::setPitch(double pitch)
{
    while(pitch < 0)
        pitch += 360;
    while(pitch >= 360)
        pitch -= 360;

    if(pitch > 70 && pitch < 290)
        return;

    this->a->setPitch(pitch);
}

void Camera::setRoll(double roll)
{
    while(roll < 0)
        roll += 360;

    while(roll >= 360)
        roll -= 360;

    this->a->setRoll(roll);
}

Camera::~Camera()
{
    try
    {
        delete a;
        delete p;
    }
    catch(...)
    {
        qDebug() << "destruction de camera -> echec";
    }
}
