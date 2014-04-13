#ifndef MOUSE_H
#define MOUSE_H
#include <QPoint>

class QMouseEvent;
class QWheelEvent;

class Mouse
    {
    public:
        static Mouse* getInstance();
        void setMousePos(QMouseEvent *e);
        int getDx();
        int getDy();
        void setDx(int dx) {this->dx = dx;}
        void setDy(int dy) {this->dy = dy;}
        bool isLeftButtonDown(){return this->isLeftButtonDown_;}
        bool isRightButtonDown(){return this->isRightButtonDown_;}
        void setButtonsDown(QMouseEvent* e);
        void setButtonsUp(QMouseEvent* e);
        void setButtonsReleased(QMouseEvent* e);
        void setScroll(QWheelEvent *e);
        bool isButtonDown(Qt::MouseButtons button);
        QPoint getLastMouseClickPos() {return this->lastMouseClickPos;}
        QPoint getLastMousePos() {return this->lastMousePos;}
        int getScrollDy() {return this->scrollDy;}
        void setScrollDy(int dy) {this->scrollDy = dy;}

    private:
        Mouse();

        QPoint lastMousePos;
        QPoint lastMouseClickPos;
        int dx;
        int dy;
        bool isLeftButtonDown_;
        bool isRightButtonDown_;
        static Mouse* instance;
        int scrollDy;
    };

#endif // MOUSE_H
