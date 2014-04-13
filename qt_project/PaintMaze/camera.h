#ifndef CAMERA_H
#define CAMERA_H

#include "angle3d.h"

class QVector3D;

class Camera
{
public:
    Camera(QVector3D *p, Angle3D *a);
    ~Camera();

    void view();

    void setYaw(double yaw);
    void setPitch(double pitch);
    void setRoll(double roll);

    void setP(QVector3D *point) { this->p = point; }

    QVector3D* getPoint();
    Angle3D* getAngle();

private:
    Angle3D *a;
    QVector3D *p;

};

#endif // CAMERA_H
