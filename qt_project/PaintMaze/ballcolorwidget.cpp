#include <QFont>
#include <QColor>
#include "player.h"
#include "ballcolorwidget.h"
#include "mainglwidget.h"

BallColorWidget::BallColorWidget(Player *player)
{
    this->player = player;
    this->p = new QVector3D(10.0, 80.0, 0.0);
}

void BallColorWidget::draw()
{
    QFont *font = new QFont("Arial");
    font->setBold(true);
    font->setPixelSize(18);

    MainGLWidget::getInstance()->renderText(this->p->x(), this->getP()->y(), "Ball color:", *font);

    QColor color = player->getCurrentColor();

    glMatrixMode(GL_PROJECTION);
    glLoadIdentity();

    int width = MainGLWidget::getInstance()->width();
    int height = MainGLWidget::getInstance()->height();

    glOrtho(0, width, height, 0, -1, 1);

    glMatrixMode(GL_MODELVIEW);
    glLoadIdentity();

    glColor3f(color.red()/255.0, color.green()/255.0, color.blue()/255.0);

    glBegin(GL_QUADS);
        glVertex2d(110.0, 85.0);
        glVertex2d(130.0, 85.0);
        glVertex2d(130.0, 65.0);
        glVertex2d(110.0, 65.0);
    glEnd();

    MainGLWidget::getInstance()->resizeGL(width, height);
}
