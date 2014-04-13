#ifndef MAINMENUSCENE_H
#define MAINMENUSCENE_H

#include <QMap>
#include <QRect>
#include "scene_a.h"

class MainMenuInputHandler;

class MainMenuScene : public Scene_A
{
public:
    MainMenuScene();
    void render();
    QMap<QString, QRect>* getMenus() {return this->menus;}

private:
    void displayMainTitle();
    void displayMainMenu();

    MainMenuInputHandler *inputHandler;
    QMap<QString, QRect> *menus;
};

#endif // MAINMENUSCENE_H
