#ifndef FACE_H
#define FACE_H

#include "QVector3D"

class Face
{
public:
    Face();
    ~Face();
    void add(QVector3D *);
    bool calculDistance(QVector3D *, double*);
    static double distanceDeuxPoints(QVector3D *p, QVector3D *pCopy);
    QVector<QVector3D*> getListFace() { return list; }
    void projectPoint(QVector3D* p);
    QVector3D * getNormal();
    QVector<QVector3D*>* getList();

private:
    QVector<QVector3D*> list;
    QVector3D *normal;
    double d;
};

#endif // FACE_H
