#include "creditsscene.h"
#include "mainglwidget.h"
#include "mouse.h"

CreditsScene::CreditsScene()
{
    this->inputHandler = new CreditsInputHandler(this);
    this->menus = new QMap<QString, QRect>();

    this->titleFont = new QFont("Arial", 76);
    this->menuFont = new QFont("Arial", 40);
    this->creditsFont = new QFont("Arial", 20);

    // add menus
    QString returnToMainMenu = "Return to main menu";

    int height = 600;
    QFontMetrics fontMetrics(*menuFont);

    int width = MainGLWidget::getInstance()->width()/2;
    int textWidth = fontMetrics.width(returnToMainMenu);
    int textHeight = fontMetrics.height();
    QRect menu1(width - textWidth/2, height-textHeight, textWidth, textHeight);

    // save menu positions
    this->menus->insert(returnToMainMenu, menu1);
}

void CreditsScene::render()
{
    this->inputHandler->handleInputs();

    displayTitle();
    displayCredits();
    displayMenu();
}

void CreditsScene::displayTitle()
{
    glColor3d(1, 1, 1);
    QString title = "Credits";
    int height = 200;

    QFontMetrics fontMetrics(*titleFont);
    int width = (MainGLWidget::getInstance()->width()/2) - fontMetrics.width(title) / 2;

    MainGLWidget::getInstance()->renderText(width, height, title, *titleFont);
}

void CreditsScene::displayCredits()
{
    glColor3d(1, 1, 1);

    // authors
    QString authors = "Nils Amiet, Raphael Capocasale, Matthieu Chavaillaz, Davy Claude, William Droz";

    int height = 300;

    QFontMetrics fontMetrics(*creditsFont);
    int width = (MainGLWidget::getInstance()->width()/2) - fontMetrics.width(authors) / 2;

    MainGLWidget::getInstance()->renderText(width, height, authors, *creditsFont);

    // copyright
    height += 80;
    QString copyright = "Copyright 2012-2013 Haute Ecole Arc";

    width = (MainGLWidget::getInstance()->width()/2) - fontMetrics.width(copyright) / 2;
    MainGLWidget::getInstance()->renderText(width, height, copyright, *creditsFont);
}

void CreditsScene::displayMenu()
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
