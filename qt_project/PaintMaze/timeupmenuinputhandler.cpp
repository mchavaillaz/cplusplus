#include <QDateTime>
#include "timeupmenuinputhandler.h"
#include "mouse.h"
#include "timeupscene.h"
#include "mainglwidget.h"
#include "mainmenuscene.h"

TimeUpMenuInputHandler::TimeUpMenuInputHandler(TimeUpScene *timeUpScene)
{
    this->timeUpScene = timeUpScene;
}

void TimeUpMenuInputHandler::handleInputs()
{
    Mouse *mouse = Mouse::getInstance();

    long timeSinceMenuInit = QDateTime::currentMSecsSinceEpoch() - this->menuInitMillis;

    if(mouse->isLeftButtonDown() && timeSinceMenuInit >= this->menuDelay)
    {
        QPoint clickPos = mouse->getLastMouseClickPos();

        foreach(QRect menuRect, this->timeUpScene->getMenus()->values())
        {
            if(menuRect.contains(clickPos.x(), clickPos.y()))
            {
                QString title = this->timeUpScene->getMenus()->key(menuRect);
                if(title == "Try again")
                {
                    MainGLWidget::getInstance()->enableLevelMode();
                }
                else if(title == "Return to main menu")
                {
                    MainGLWidget::getInstance()->setCurrentScene(new MainMenuScene());
                }
            }
        }
    }
}
