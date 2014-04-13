#ifndef PLAYER_H
#define PLAYER_H

#include <QString>
#include <QList>
#include <QColor>
#include "collidable_a.h"

class Camera;
class Level_A;
class Ball;

class Player : public Collidable_A
{
    Q_OBJECT

public:
    Player(QVector3D *p, Camera *camera, Level_A *level, QString name, double speed=0.15, int ammo=600, double mouseSpeed=0.1);
    bool isColliding(Collidable_A *);
    void draw();
    void actionGoStraight();
    void actionGoBackwards();
    void actionStrafeRight();
    void actionStrafeLeft();
    void actionTirer();
    void actionNextColor();
    void actionPreviousColor();
    void actionDiagUpRight();
    void actionDiagUpLeft();
    void actionDiagDownRight();
    void actionDiagDownLeft();
    void actionLook(double dx, double dy);
    void slide(double xSlide, double zSlide);

    void createStain(QVector3D *p, QVector3D*, Ball *ball, double angle = 90);

    int getAmmo(){ return ammo; }
    QColor getCurrentColor() { return this->listColor.at(this->currentColorIndex); }
    void nextColor();
    void previousColor();

private:
    void move(double speed, double angleDifference=0);
    QString name;
    Camera *camera;
    double speed;
    int ammo;
    QList<QColor> listColor;
    int currentColorIndex;
    double mouseSpeed;
    Level_A *level;
    int lastShotTime;
    int shotInterval;
};

#endif // PLAYER_H
