#ifndef LEVELFINISHEDSCENE_H
#define LEVELFINISHEDSCENE_H

#include <QFont>
#include <QMap>
#include <QRect>
#include "scene_a.h"
#include "levelfinishedinputhandler.h"

class LevelFinishedScene : public Scene_A
{
public:
    LevelFinishedScene(long timeLeftMillis, int ammoLeft);
    void render();
    QMap<QString, QRect>* getMenus() { return this->menus; }

private:
    LevelFinishedInputHandler *inputHandler;
    QMap<QString, QRect> *menus;

    QFont *titleFont;
    QFont *menuFont;
    QFont *scoreFont;

    QString timeLeft;
    int ammoLeft;

    void displayTitle();
    void displayScore();
    void displayMenu();
};

#endif // LEVELFINISHEDSCENE_H
