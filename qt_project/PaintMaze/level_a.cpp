#include <QDateTime>
#include "level_a.h"
#include "shape_a.h"
#include "endobject.h"

Level_A::Level_A()
{
    this->timeLimitMillis = 2 * 60 * 1000; //2 minutes default
}

QList<Collidable_A *> Level_A::getListShapeCollisionnable()
{
    return listShapeCollidable;
}

QList<Shape_A *> Level_A::getListShapeNonCollisionnable()
{
    return listShapeNonCollidable;
}

Level_A::~Level_A()
{
    listShapeCollidable.clear();
    listShapeNonCollidable.clear();
}

void Level_A::addMovable(Movable *a)
{
    this->listMovable.append(a);
}

QList<Movable *> Level_A::getListMovable()
{
    return this->listMovable;
}

void Level_A::removeAndDeleteMovable(Movable *p)
{
    this->listMovable.removeAll(p);
}

void Level_A::start()
{
    this->startTimeMillis = QDateTime::currentMSecsSinceEpoch();
}

long Level_A::getTimeLeftMillis()
{
    return this->timeLimitMillis - (QDateTime::currentMSecsSinceEpoch() - this->startTimeMillis);
}
