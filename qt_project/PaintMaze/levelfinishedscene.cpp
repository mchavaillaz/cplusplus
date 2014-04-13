#include <QDateTime>
#include "levelfinishedscene.h"
#include "mainglwidget.h"
#include "mouse.h"
#include "levelfinishedinputhandler.h"

LevelFinishedScene::LevelFinishedScene(long timeLeftMillis, int ammoLeft)
{
    this->timeLeft = "Time left: ";

    QDateTime dateTime = QDateTime::fromTime_t(0);
    dateTime = dateTime.addMSecs(timeLeftMillis);

    this->timeLeft += dateTime.toString("mm:ss:zzz");

    this->ammoLeft = ammoLeft;

    this->titleFont = new QFont("Arial", 76);
    this->menuFont = new QFont("Arial", 40);
    this->scoreFont = new QFont("Arial", 20);

    this->inputHandler = new LevelFinishedInputHandler(this);
    this->menus = new QMap<QString, QRect>();

    // add menus
    QString replay = "Replay";
    QString returnToMainMenu = "Return to main menu";

    int height = 600;
    QFontMetrics fontMetrics(*menuFont);

    int width = MainGLWidget::getInstance()->width()/2;
    int textWidth = fontMetrics.width(replay);
    int textHeight = fontMetrics.height();
    QRect menu1(width - textWidth/2, height-textHeight, textWidth, textHeight);

    // Return to main menu
    height += 80;
    textWidth = fontMetrics.width(returnToMainMenu);
    QRect menu2(width - textWidth/2, height-textHeight, textWidth, textHeight);

    // save menu positions
    this->menus->insert(replay, menu1);
    this->menus->insert(returnToMainMenu, menu2);
}

void LevelFinishedScene::render()
{
    this->inputHandler->handleInputs();

    displayTitle();
    displayScore();
    displayMenu();
}

void LevelFinishedScene::displayTitle()
{
    glColor3d(1, 0, 0);
    QString title = "Level completed!";
    int height = 200;

    QFontMetrics fontMetrics(*titleFont);
    int width = (MainGLWidget::getInstance()->width()/2) - fontMetrics.width(title) / 2;

    MainGLWidget::getInstance()->renderText(width, height, title, *titleFont);
}

void LevelFinishedScene::displayScore()
{
    glColor3d(1, 1, 1);
    QString ammoLeft = "Ammo left: " + QString::number(this->ammoLeft);
    int height = 300;

    QFontMetrics fontMetrics(*scoreFont);
    int width = (MainGLWidget::getInstance()->width()/2) - fontMetrics.width(this->timeLeft) / 2;

    MainGLWidget::getInstance()->renderText(width, height, this->timeLeft, *scoreFont);

    height += 50;
    width = (MainGLWidget::getInstance()->width()/2) - fontMetrics.width(ammoLeft) / 2;

    MainGLWidget::getInstance()->renderText(width, height, ammoLeft, *scoreFont);
}

void LevelFinishedScene::displayMenu()
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
