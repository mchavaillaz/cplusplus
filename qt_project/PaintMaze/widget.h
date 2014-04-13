#ifndef WIDGET_H
#define WIDGET_H

#include <QFont>
#include <QVector3D>
#include "shape_a.h"

class Widget : public Shape_A
{
public:
    Widget(QVector3D *p = new QVector3D(0,0,0));
    virtual void draw()=0;
};

#endif // WIDGET_H
