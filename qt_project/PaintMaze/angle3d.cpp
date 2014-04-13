#include "angle3d.h"

Angle3D::Angle3D(double _yaw, double _pitch, double _roll) : yaw(_yaw), pitch(_pitch), roll(_roll)
{
}

Angle3D::~Angle3D() {

}

Angle3D::Angle3D(const Angle3D &copy)
{
    this->yaw = copy.getYaw();
    this->pitch = copy.getPitch();
    this->roll = copy.getRoll();
}

Angle3D& Angle3D::operator=(const Angle3D &obj)
{
    if(this != &obj)
    {
        this->yaw = obj.getYaw();
        this->pitch = obj.getPitch();
        this->roll = obj.getRoll();
    }
    return *this;
}

double Angle3D::getYaw() const
{
    return this->yaw;
}

double Angle3D::getPitch() const
{
    return this->pitch;
}

double Angle3D::getRoll() const
{
    return this->roll;
}

void Angle3D::setYaw(double _yaw)
{
    this->yaw = _yaw;
}

void Angle3D::setPitch(double _pitch)
{
    this->pitch = _pitch;
}

void Angle3D::setRoll(double _roll)
{
    this->roll = _roll;
}

void Angle3D::addYaw(double _yaw)
{
    this->yaw += _yaw;
}

void Angle3D::addPitch(double _pitch)
{
    this->pitch += _pitch;
}

void Angle3D::addRoll(double _roll)
{
    this->roll += _roll;
}
