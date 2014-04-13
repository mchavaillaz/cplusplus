#ifndef AMMOWIDGET_H
#define AMMOWIDGET_H

#include <QFont>
#include "widget.h"

class Player;

class AmmoWidget : public Widget
{
public:
    AmmoWidget(Player *player);
    void draw();

private:
    QFont font;
    Player *player;
};

#endif // AMMOWIDGET_H
