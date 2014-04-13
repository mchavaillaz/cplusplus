#ifndef MAINMENUUNPUTHANDLER_H
#define MAINMENUUNPUTHANDLER_H

#include "inputhandler_a.h"
#include "mainmenuscene.h"

class MainMenuInputHandler : public InputHandler_A
{
public:
    MainMenuInputHandler(MainMenuScene*);
    void handleInputs();

private:
    MainMenuScene *mainMenuScene;
};

#endif // MAINMENUUNPUTHANDLER_H
