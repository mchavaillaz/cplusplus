#ifndef TIMEWIDGET_H
#define TIMEWIDGET_H

#include "widget.h"
#include "level_a.h"

class TimeWidget: public Widget
{
public:
    TimeWidget(Level_A *level);
    void draw();

 private:
    QFont font;
    Level_A *level;
};

#endif // TIMEWIDGET_H
