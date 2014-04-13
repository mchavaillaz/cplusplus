#ifndef BALLCOLORWIDGET_H
#define BALLCOLORWIDGET_H

#include "widget.h"

class Player;

class BallColorWidget: public Widget
{
public:
    BallColorWidget(Player *player);
    void draw();
private:
    Player *player;

};

#endif // BALLCOLORWIDGET_H
