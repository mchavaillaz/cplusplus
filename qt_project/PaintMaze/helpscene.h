#ifndef HELPSCENE_H
#define HELPSCENE_H

#include "scene_a.h"
#include "helpinputhandler.h"

class HelpScene : public Scene_A
{
public:
    HelpScene();
    void render();
    QMap<QString, QRect>* getMenus() { return this->menus; }
    static bool boolSwitch;

private:
    QFont *menuFont;
    QFont *titleFont;

    QImage *image;
    QImage imageFini;

    GLuint texture[20];
    HelpInputHandler *inputHandler;
    QMap<QString, QRect> *menus;


    void displayImage();
    void displayMenu();
    void displayTitle();
};

#endif // HELPSCENE_H
