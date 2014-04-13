#include "timeupscene.h"
#include "mainglwidget.h"
#include "mouse.h"
#include "timeupmenuinputhandler.h"

TimeUpScene::TimeUpScene()
{
    this->timeUpFont = new QFont("Arial", 76);
    this->menuFont = new QFont("Arial", 40);

    this->inputHandler = new TimeUpMenuInputHandler(this);
    this->menus = new QMap<QString, QRect>();

    // add menus
    QString tryAgain = "Try again";
    QString returnToMainMenu = "Return to main menu";

    int height = 450;
    QFontMetrics fontMetrics(*menuFont);

    // Try again
    int width = MainGLWidget::getInstance()->width()/2;
    int textWidth = fontMetrics.width(tryAgain);
    int textHeight = fontMetrics.height();
    QRect tryAgaintRect(width - textWidth/2, height-textHeight, textWidth, textHeight);

    // Return to main menu
    height += 80;
    textWidth = fontMetrics.width(returnToMainMenu);
    QRect returnTomainMenuRect(width - textWidth/2, height-textHeight, textWidth, textHeight);

    // save menu positions
    this->menus->insert(tryAgain, tryAgaintRect);
    this->menus->insert(returnToMainMenu, returnTomainMenuRect);
}

void TimeUpScene::render()
{
    this->inputHandler->handleInputs();
    displayTimeUp();
    displayTimeUpMenu();
}

void TimeUpScene::displayTimeUp()
{
    glColor3d(1, 0, 0);
    QString title = "Time up!";
    int height = 200;

    QFontMetrics fontMetrics(*timeUpFont);
    int width = (MainGLWidget::getInstance()->width()/2) - fontMetrics.width(title) / 2;

    MainGLWidget::getInstance()->renderText(width, height, title, *timeUpFont);
}

void TimeUpScene::displayTimeUpMenu()
{
    glColor3d(1, 1, 1);

    QFontMetrics fontMetrics(*menuFont);

    foreach(QString menuTitle, this->menus->keys())
    {
        QRect menuRect = this->menus->value(menuTitle);

        QPoint mousePos = Mouse::getInstance()->getLastMousePos();
        if(menuRect.contains(mousePos))
        {
            glColor3d(1, 1, 0);
        }
        else
        {
            glColor3d(1, 1, 1);
        }
        MainGLWidget::getInstance()->renderText(menuRect.left(), menuRect.bottom(), menuTitle, *menuFont);
    }
}
