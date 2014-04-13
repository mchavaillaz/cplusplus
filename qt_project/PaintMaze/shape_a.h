#ifndef SHAPE_A_H
#define SHAPE_A_H

#include <QObject>

#include <QVector3D>


class Shape_A : public QObject
{
    Q_OBJECT
public:
    Shape_A(QVector3D *p);
    Shape_A(const Shape_A &copy);
    Shape_A& operator=(const Shape_A &);

    virtual ~Shape_A();

    virtual void draw() = 0;

    void setP(QVector3D*p);
    QVector3D* getP() const;

public:
    QVector3D *p;
};

#endif // SHAPE_A_H
