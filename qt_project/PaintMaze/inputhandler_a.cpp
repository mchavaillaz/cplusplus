#include <QDateTime>
#include "inputhandler_a.h"

InputHandler_A::InputHandler_A()
{
    this->menuDelay = 600;
    this->menuInitMillis = QDateTime::currentMSecsSinceEpoch();
}
