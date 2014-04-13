#ifndef LEVELFINISHEDINPUTHANDLER_H
#define LEVELFINISHEDINPUTHANDLER_H

#include "inputhandler_a.h"


class LevelFinishedScene;

class LevelFinishedInputHandler : public InputHandler_A
{
public:
    LevelFinishedInputHandler(LevelFinishedScene *scene);
    void handleInputs();

private:
    LevelFinishedScene *scene;
};

#endif // LEVELFINISHEDINPUTHANDLER_H
