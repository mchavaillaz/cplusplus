#include <QDateTime>
#include "helpinputhandler.h"
#include "mouse.h"
#include "mainglwidget.h"
#include "mainmenuscene.h"
#include "helpscene.h"

HelpInputHandler::HelpInputHandler(HelpScene *scene)
{
    this->scene = scene;
}

void HelpInputHandler::handleInputs()
{
    Mouse *mouse = Mouse::getInstance();

    long timeSinceMenuInit = QDateTime::currentMSecsSinceEpoch() - this->menuInitMillis;

    if(mouse->isLeftButtonDown() && timeSinceMenuInit >= this->menuDelay)
    {
        QPoint clickPos = mouse->getLastMouseClickPos();

        foreach(QRect menuRect, this->scene->getMenus()->values())
        {
            if(menuRect.contains(clickPos.x(), clickPos.y()))
            {
                QString title = this->scene->getMenus()->key(menuRect);
                if(title == "Return to main menu")
                {
                    MainGLWidget::getInstance()->setCurrentScene(new MainMenuScene());
                }
                else if(title == "Next")
                {
                    HelpScene::boolSwitch ^= true;
                    MainGLWidget::getInstance()->setCurrentScene(new HelpScene());
                }
            }
        }
    }
}
