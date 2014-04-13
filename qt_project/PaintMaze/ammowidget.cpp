#include <QVector>
#include "player.h"
#include "ammowidget.h"
#include "mainglwidget.h"

AmmoWidget::AmmoWidget(Player *player) : Widget(new QVector3D(10,20,0))
{
    this->player = player;
    this->font.setFamily("Arial");
    this->font.setBold(true);
    this->font.setPixelSize(18);
}

void AmmoWidget::draw()
{
    glColor3d(1, 1, 1);
    MainGLWidget::getInstance()->renderText(this->p->x(), this->p->y(), "Ammo: " + QString::number(player->getAmmo()), font);
}
