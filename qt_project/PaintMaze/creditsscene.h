#ifndef CREDITSSCENE_H
#define CREDITSSCENE_H

#include "scene_a.h"
#include "creditsinputhandler.h"

class CreditsScene : public Scene_A
{
public:
    CreditsScene();
    void render();
    QMap<QString, QRect>* getMenus() { return this->menus; }

private:
    CreditsInputHandler *inputHandler;
    QMap<QString, QRect> *menus;

    QFont *titleFont;
    QFont *menuFont;
    QFont *creditsFont;

    void displayTitle();
    void displayCredits();
    void displayMenu();
};

#endif // CREDITSSCENE_H
