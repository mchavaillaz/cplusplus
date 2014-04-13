#include <QDateTime>
#include "mainmenuinputhandler.h"
#include "mouse.h"
#include "mainglwidget.h"
#include "creditsscene.h"
#include "helpscene.h"

MainMenuInputHandler::MainMenuInputHandler(MainMenuScene *mainMenuScene)
{
    this->mainMenuScene = mainMenuScene;
}

void MainMenuInputHandler::handleInputs()
{
    Mouse *mouse = Mouse::getInstance();

    long timeSinceMenuInit = QDateTime::currentMSecsSinceEpoch() - this->menuInitMillis;

    if(mouse->isLeftButtonDown() && timeSinceMenuInit >= this->menuDelay)
    {
        QPoint clickPos = mouse->getLastMouseClickPos();

        foreach(QRect menuRect, this->mainMenuScene->getMenus()->values()) {
            if(menuRect.contains(clickPos.x(), clickPos.y()))
            {
                QString title = this->mainMenuScene->getMenus()->key(menuRect);
                if(title == "Exit")
                {
                    exit(0);
                }
                else if(title == "Start")
                {
                    MainGLWidget::getInstance()->enableLevelMode();
                }
                else if(title == "Help")
                {
                    MainGLWidget::getInstance()->setCurrentScene(new HelpScene());
                }
                else if(title == "Credits")
                {
                    MainGLWidget::getInstance()->setCurrentScene(new CreditsScene());
                }
            }
        }
    }
}
