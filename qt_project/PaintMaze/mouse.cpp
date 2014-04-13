#include <QMouseEvent>
#include "mouse.h"
#include "mainglwidget.h"

Mouse* Mouse::instance = 0;

Mouse* Mouse::getInstance()
{
    if(Mouse::instance == 0)
    {
        Mouse::instance = new Mouse();
    }

    return Mouse::instance;
}

Mouse::Mouse()
{
    this->isLeftButtonDown_ = false;
}

int Mouse::getDx() {
    int dxTemp = this->dx;
    this->dx = 0;
    return dxTemp;
}

int Mouse::getDy() {
    int dyTemp = this->dy;
    this->dy = 0;
    return dyTemp;
}

void Mouse::setMousePos(QMouseEvent *e)
{
    this->dx = -this->lastMousePos.x() + e->x();
    this->dy = -this->lastMousePos.y() + e->y();

    if(abs(this->dx) <= 1) {
        this->dx = 0;
    }
    if(abs(this->dy) <= 1) {
        this->dy = 0;
    }

    this->lastMousePos = QPoint(e->x(), e->y());
}

void Mouse::setButtonsDown(QMouseEvent* e)
{
    if(e->buttons() & Qt::RightButton)
    {
        this->isRightButtonDown_ = true;
    }
    if(e->buttons() & Qt::LeftButton)
    {
        this->isLeftButtonDown_ = true;
    }

    this->lastMouseClickPos = e->pos();
}

void Mouse::setButtonsUp(QMouseEvent *e)
{
    if(e->button() == Qt::RightButton)
    {
        this->isRightButtonDown_ = false;
    }
    if(e->button() == Qt::LeftButton)
    {
        this->isLeftButtonDown_ = false;
    }
}

void Mouse::setButtonsReleased(QMouseEvent* e)
{
    if(e->button() == Qt::RightButton)
    {
        this->isRightButtonDown_ = false;
    }
    if(e->button() == Qt::LeftButton)
    {
        this->isLeftButtonDown_ = false;
    }
}

bool Mouse::isButtonDown(Qt::MouseButtons button)
{
    if(button == Qt::LeftButton)
    {
        return this->isLeftButtonDown_;
    }
    else if(button == Qt::RightButton)
    {
        return this->isRightButtonDown_;
    }

    return false;
}

void Mouse::setScroll(QWheelEvent *e)
{
    this->scrollDy = e->delta();
}
