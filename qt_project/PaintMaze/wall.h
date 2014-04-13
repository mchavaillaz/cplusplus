#ifndef WALL_H
#define WALL_H

#include "collidable_a.h"
#include "QGLWidget"

class QGLBuffer;
class QVector3D;
class Face;
class Ball;

class Wall : public Collidable_A
{
    Q_OBJECT
public:
    Wall(QVector3D *p1, QVector3D *p2, QVector3D *p3, QVector3D *p4);
    ~Wall();

    void draw();
    bool isColliding(Collidable_A *);
    void createStain(QVector3D *p,QVector3D*, Ball *ball, double angle = 90);
    QVector3D* getP3() { return this->p3; }

private:
    QVector3D *p2;
    QVector3D *p3;
    QVector3D *p4;

    QGLBuffer *buffer;
    QGLBuffer *buffer2;

    QList<Face*> listFace;

    int bufferID;
    static double height;
    void addFace(QVector3D *p1, QVector3D *p2, QVector3D *p3, QVector3D *p4);
};

#endif // WALL_H
