#include <cmath>
#include <QDateTime>
#include <QVector3D>
#include "level_a.h"
#include "player.h"
#include "collidable_a.h"
#include "camera.h"
#include "ball.h"


Player::Player(QVector3D *p, Camera *camera, Level_A *level, QString name, double speed, int ammo, double mouseSpeed) : Collidable_A(p), name(name), listColor()
{
    this->camera = camera;
    this->speed = speed;
    this->ammo = ammo;
    this->listColor.append(Qt::red);
    this->listColor.append(Qt::green);
    this->listColor.append(Qt::blue);
    this->currentColorIndex = 0;
    this->level = level;
    this->mouseSpeed = mouseSpeed;
    this->lastShotTime = QDateTime::currentMSecsSinceEpoch();
    this->shotInterval = 150;
}

void Player::createStain(QVector3D *p,QVector3D*, Ball *ball, double angle)
{

}

bool Player::isColliding(Collidable_A *objet)
{
    return false;
}

void Player::draw()
{
// rien
}

void Player::slide(double xSlide, double zSlide)
{
    this->p->setX(xSlide);
    this->p->setZ(zSlide);

    this->camera->getPoint()->setX(this->p->x());
    this->camera->getPoint()->setZ(this->p->z());
}

void Player::move(double speed, double angleDifference) { // angleDifference=0
    double angle = angleDifference - camera->getAngle()->getYaw();

    angle = angle * M_PI / 180.0;
    double dx = sin(angle) * speed;
    double dz = cos(angle) * speed;

    this->p->setX(this->p->x() + dx);
    this->p->setZ(this->p->z() + dz);

    this->camera->getPoint()->setX(this->p->x());
    this->camera->getPoint()->setZ(this->p->z());
}

void Player::actionDiagUpLeft() {
    move(this->speed, 45);
}

void Player::actionDiagUpRight() {
    move(this->speed, -45);
}

void Player::actionDiagDownLeft() {
    move(-this->speed, -45);
}

void Player::actionDiagDownRight() {
    move(-this->speed, 45);
}

void Player::actionGoStraight()
{
    move(this->speed);
}

void Player::actionGoBackwards()
{
    move(-this->speed);
}

void Player::actionStrafeRight()
{
    move(this->speed, -90);
}

void Player::actionStrafeLeft()
{
    move(this->speed, 90);
}

void Player::actionTirer()
{
    int timeSinceLastShot = QDateTime::currentMSecsSinceEpoch() - lastShotTime;
    if(this->ammo > 0 && timeSinceLastShot > shotInterval)
    {
        this->lastShotTime = QDateTime::currentMSecsSinceEpoch();
        this->ammo--;
        this->level->addMovable(new Ball(this->p, new Angle3D(*(this->camera->getAngle())), this->getCurrentColor()));
    }
}

void Player::actionLook(double dx, double dy)
{
    this->camera->setYaw(this->camera->getAngle()->getYaw()-dx*this->mouseSpeed);
    this->camera->setPitch(this->camera->getAngle()->getPitch()-dy*this->mouseSpeed);
}

void Player::actionNextColor()
{
    this->currentColorIndex = (this->currentColorIndex+1) % this->listColor.size();
}

void Player::actionPreviousColor()
{
    this->currentColorIndex = (this->currentColorIndex-1) % this->listColor.size();
}

void Player::nextColor()
{
    this->currentColorIndex = (this->currentColorIndex + 1) % this->listColor.size();
}

void Player::previousColor()
{
    this->currentColorIndex--;
    if(this->currentColorIndex < 0)
    {
        this->currentColorIndex = this->listColor.size() - 1;
    }
}
