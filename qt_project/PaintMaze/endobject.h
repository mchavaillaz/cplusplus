#ifndef ENDOBJECT_H
#define ENDOBJECT_H

#include "collidable_a.h"
#include "QImage"


class EndObject : public Collidable_A
{
public:
    EndObject(QVector3D *);
    void draw();
    bool isColliding(Collidable_A *);
    void createStain(QVector3D *p, QVector3D*, Ball *ball, double angle = 0);
    void createTextEnd(QVector3D *);


private:
    void createEndObject();
    void createZoneEnd();
    QImage image;
    unsigned int texID;
    double spacing;
    double height;
};

#endif // ENDOBJECT_H
