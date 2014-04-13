#ifndef LEVEL_A_H
#define LEVEL_A_H

#include <QList>
#include <QGLWidget>
#include <GL/glu.h>

class Player;
class Collidable_A;
class Shape_A;
class QVector3D;
class Movable;
class EndObject;

class Level_A
{
public:
    Level_A();
    virtual ~Level_A();

    QList<Collidable_A *> getListShapeCollisionnable();
    QList<Shape_A *> getListShapeNonCollisionnable();
    QList<Movable *> getListMovable();
    QVector3D * getStartingPoint() { return this->startingPoint; }
    QVector3D * getGoalPoint() { return this->goalPoint; }
    EndObject* getEndObject() { return this->endObject; }
    long getTimeLeftMillis();
    void addMovable(Movable *);
    void removeAndDeleteMovable(Movable *);
    void start();

protected:
    Player *player;
    QList<Collidable_A *> listShapeCollidable;
    QList<Shape_A *> listShapeNonCollidable;
    QList<Movable *> listMovable;
    QVector3D *startingPoint;
    QVector3D *goalPoint;
    EndObject *endObject;

    long timeLimitMillis;
    long startTimeMillis;
};

#endif // LEVEL_A_H
