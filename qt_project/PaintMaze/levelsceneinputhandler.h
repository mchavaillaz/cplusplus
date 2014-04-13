#ifndef LEVELSCENEINPUTHANDLER_H
#define LEVELSCENEINPUTHANDLER_H

#include <QPoint>
#include <QList>
#include "inputhandler_a.h"

class Player;
class Keyboard;
class Mouse;
class Level_A;
class Collidable_A;
class Camera;

class LevelSceneInputHandler : public InputHandler_A
{
public:
    LevelSceneInputHandler(Player*, QPoint *point, Level_A *level, Camera *camera);
    void handleInputs();

private:
    Player *player;
    Player *playerCopy;
    Player *playerCopyX;
    Player *playerCopyZ;
    Camera *cameraCopy;
    Camera *camera;
    Keyboard *keyboard;
    Mouse *mouse;
    Level_A *level;
    QList<Collidable_A *> listCollidableCopy;

    bool isColliding(Player *playerCopy);
};

#endif // LEVELSCENEINPUTHANDLER_H
