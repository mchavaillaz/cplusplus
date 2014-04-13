#include <QGLWidget>
#include "helpscene.h"
#include "helpinputhandler.h"
#include "mainglwidget.h"
#include "mouse.h"
#include <QDebug>
#include <exception>

bool HelpScene::boolSwitch = true;

HelpScene::HelpScene()
{
    this->menuFont = new QFont("Arial", 40);

    this->inputHandler = new HelpInputHandler(this);
    this->menus = new QMap<QString, QRect>();
    this->titleFont = new QFont("Arial", 76);

    // add menus
    QString replay = "Next";
    QString returnToMainMenu = "Return to main menu";
    QString path ="image/";

    //image = 0;

    if(boolSwitch)
        image = new QImage(path + "keyboard.png");
    else
        image = new QImage(path + "game.png");


    int height = 750;

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

    imageFini = QImage(QGLWidget::convertToGLFormat( *image ));
    delete image;
    glGenTextures( 1, &texture[0] );
    glBindTexture( GL_TEXTURE_2D, texture[0] );
    glTexImage2D( GL_TEXTURE_2D, 0, 3, imageFini.width(), imageFini.height(), 0, GL_RGBA, GL_UNSIGNED_BYTE, imageFini.bits() );
    glTexParameteri( GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR );
    glTexParameteri( GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR );

    glEnable(GL_TEXTURE_2D);
}

void HelpScene::render()
{
    this->inputHandler->handleInputs();
    displayImage();
    displayMenu();
    displayTitle();
}


void HelpScene::displayImage()
{

      glMatrixMode(GL_PROJECTION);
      glLoadIdentity();

      int width = MainGLWidget::getInstance()->width();
      int height = MainGLWidget::getInstance()->height();

      glOrtho(0, width, height, 0, -1, 1);

      glMatrixMode(GL_MODELVIEW);
      glLoadIdentity();

      glBegin(GL_QUADS);

                       glTexCoord2i(1,0);
      glVertex2i(width/2 + imageFini.width()/3, height/2 + imageFini.height()/3);
      glTexCoord2i(0,0);

      glVertex2i(width/2 - imageFini.width()/3, height/2 + imageFini.height()/3);

      glTexCoord2i(0,1);
      glVertex2i(width/2 - imageFini.width()/3, height/2 - imageFini.height()/3);


      glTexCoord2i(1,1);
      glVertex2i(width/2 + imageFini.width()/3, height/2 - imageFini.height()/3);

      glEnd();

      MainGLWidget::getInstance()->resizeGL(width, height);
}

void HelpScene::displayTitle()
{
    glColor3d(1, 1, 1);
    QString title = "Help";
    int height = 200;

    QFontMetrics fontMetrics(*titleFont);
    int width = (MainGLWidget::getInstance()->width()/2) - fontMetrics.width(title) / 2;

    MainGLWidget::getInstance()->renderText(width, height, title, *titleFont);
}


void HelpScene::displayMenu()
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
