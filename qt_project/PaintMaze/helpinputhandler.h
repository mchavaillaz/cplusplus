#ifndef HELPINPUTHANDLER_H
#define HELPINPUTHANDLER_H

#include "inputhandler_a.h"

class HelpScene;

class HelpInputHandler : public InputHandler_A
{
public:
    HelpInputHandler(HelpScene *scene);
    void handleInputs();

private:
    HelpScene *scene;
};

#endif // HELPINPUTHANDLER_H
