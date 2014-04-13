#ifndef TIMEUPMENUINPUTHANDLER_H
#define TIMEUPMENUINPUTHANDLER_H

#include "inputhandler_a.h"
#include "timeupscene.h"

class TimeUpMenuInputHandler : public InputHandler_A
{
public:
    TimeUpMenuInputHandler(TimeUpScene*);
    void handleInputs();

private:
    TimeUpScene *timeUpScene;
};

#endif // TIMEUPMENUINPUTHANDLER_H
