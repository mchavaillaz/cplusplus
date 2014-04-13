#include <QFont>
#include <QGLWidget>
#include <QFontMetrics>
#include "mainmenuscene.h"
#include "mainglwidget.h"
#include "mainmenuinputhandler.h"
#include "mouse.h"

MainMenuScene::MainMenuScene()
{
    this->inputHandler = new MainMenuInputHandler(this);
    this->menus = new QMap<QString, QRect>();

    // add menus
    QString start = "Start";
    QString help = "Help";
    QString credits = "Credits";
    QString exit = "Exit";

    int height = 450;
    QFont mainMenuFont("Arial", 40);
    QFontMetrics fontMetrics(mainMenuFont);

    // start
    int width = MainGLWidget::getInstance()->width()/2;
    int textWidth = fontMetrics.width(start);
    int textHeight = fontMetrics.height();
    QRect startRect(width - textWidth/2, height-textHeight, textWidth, textHeight);

    // help
    height += 80;
    textWidth = fontMetrics.width(help);
    QRect helpRect(width - textWidth/2, height-textHeight, textWidth, textHeight);

    // credits
    height += 80;
    textWidth = fontMetrics.width(credits);
    QRect creditsRect(width - textWidth/2, height-textHeight, textWidth, textHeight);

    // exit
    height += 80;
    textWidth = fontMetrics.width(exit);
    QRect exitRect(width - textWidth/2, height-textHeight, textWidth, textHeight);

    // save menu positions
    this->menus->insert(start, startRect);
    this->menus->insert(help, helpRect);
    this->menus->insert(credits, creditsRect);
    this->menus->insert(exit, exitRect);
}

void MainMenuScene::render()
{
    glClearColor(0, 0, 0, 0);
    displayMainTitle();
    displayMainMenu();

    this->inputHandler->handleInputs();
}

void MainMenuScene::displayMainTitle()
{
    glColor3d(1, 0, 0);
    QString title = "Paint Maze";
    int height = 200;

    // main menu title font
    QFont mainTitle("Arial", 76);
    QFontMetrics fontMetrics(mainTitle);
    int width = (MainGLWidget::getInstance()->width()/2) - fontMetrics.width(title) / 2;
    // draw main title
    MainGLWidget::getInstance()->renderText(width, height, title, mainTitle);
}

void MainMenuScene::displayMainMenu()
{
    glColor3d(1, 1, 1);

    // main menu title font
    QFont mainMenuFont("Arial", 40);
    QFontMetrics fontMetrics(mainMenuFont);

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
        MainGLWidget::getInstance()->renderText(menuRect.left(), menuRect.bottom(), menuTitle, mainMenuFont);
    }
}
