#include <QDateTime>
#include "timewidget.h"
#include "mainglwidget.h"


TimeWidget::TimeWidget(Level_A *level)
{
    this->p = new QVector3D(10, 50, 0);
    this->level = level;

    this->font.setFamily("Arial");
    this->font.setBold(true);
    this->font.setPixelSize(18);
}

void TimeWidget::draw()
{
    glColor3d(1, 1, 1);

    QDateTime qDateTime = QDateTime::fromTime_t(0);
    qDateTime = qDateTime.addMSecs(this->level->getTimeLeftMillis());

    MainGLWidget::getInstance()->renderText(this->p->x(), this->p->y(), "Time left: " + qDateTime.toString("mm:ss:zzz"), font);
}
