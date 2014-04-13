#include <QGLWidget>
#include <QVector3D>
#include "util.h"
#include "face.h"

/**
 * @brief Util::sendNewPoint
 * @param pGiven point sur la tache
 * @param face face qui est en collision
 * @return true si pGiven est dans la face, false sinon
 */
bool Util::isPointInSideFace(QVector3D *pGiven, Face *face)
{
    QVector<QVector3D*> facePoints = face->getListFace();
    QVector3D * a = facePoints[0];
    QVector3D * b = facePoints[1];
    QVector3D * c = facePoints[2];
    QVector3D * d = facePoints[3];

    return sameSide(pGiven, a, b ,c) && sameSide(pGiven, b, c ,d) && sameSide(pGiven, c, d ,a) && sameSide(pGiven, d, a ,b);
}

bool Util::sameSide(QVector3D *p, QVector3D *a, QVector3D *b, QVector3D *c)
{
    QVector3D *u = new QVector3D(QVector3D::crossProduct((*c - *b), (*p - *b)));
    QVector3D *v = new QVector3D(QVector3D::crossProduct((*c - *b), (*a - *b)));

    return QVector3D::dotProduct(*u, *v) >= 0;
}

void Util::debugLine(QVector3D *a, QVector3D *b)
{
    glColor3f(1,0,0);
    glBegin(GL_LINES);
        glVertex3f(a->x(), a->y(),a->z());
        glVertex3f(b->x(), b->y(),b->z());
    glEnd();
}
