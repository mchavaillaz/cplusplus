#ifndef TIMEUPSCENE_H
#define TIMEUPSCENE_H

#include <QFont>
#include <QMap>
#include <QRect>
#include "scene_a.h"

class TimeUpMenuInputHandler;

class TimeUpScene : public Scene_A
{
public:
    TimeUpScene();
    void render();
    QMap<QString, QRect>* getMenus() { return this->menus; }

private:
    void displayTimeUp();
    void displayTimeUpMenu();

    TimeUpMenuInputHandler *inputHandler;
    QMap<QString, QRect> *menus;

    QFont *timeUpFont;
    QFont *menuFont;
};

#endif // TIMEUPSCENE_H
