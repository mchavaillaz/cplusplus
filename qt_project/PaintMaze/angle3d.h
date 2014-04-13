#ifndef ANGLE3D_H
#define ANGLE3D_H

class Angle3D
{
public:
    Angle3D(double yaw = 0, double pitch = 0, double roll = 0);
    Angle3D(const Angle3D &copie);
    Angle3D& operator=(const Angle3D &obj);
    ~Angle3D();

    double getYaw() const;
    double getPitch() const;
    double getRoll() const;

    void setYaw(double yaw);
    void setPitch(double pitch);
    void setRoll(double roll);

    void addYaw(double yaw);
    void addPitch(double pitch);
    void addRoll(double roll);

private:
    double yaw;
    double pitch;
    double roll;
};

#endif // ANGLE3D_H
