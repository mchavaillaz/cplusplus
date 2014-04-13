#ifndef INPUTHANDLER_A_H
#define INPUTHANDLER_A_H

class InputHandler_A
{
public:
    InputHandler_A();
    virtual void handleInputs() = 0;

protected:
    long menuInitMillis;
    long menuDelay;
};

#endif // INPUTHANDLER_A_H
