#ifndef CREDITSINPUTHANDLER_H
#define CREDITSINPUTHANDLER_H

#include "inputhandler_a.h"

class CreditsScene;

class CreditsInputHandler : public InputHandler_A
{
public:
    CreditsInputHandler(CreditsScene *scene);
    void handleInputs();

private:
    CreditsScene *scene;
};

#endif // CREDITSINPUTHANDLER_H
