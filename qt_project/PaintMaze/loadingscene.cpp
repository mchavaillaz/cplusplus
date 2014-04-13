#include <QFontMetrics>
#include "loadingscene.h"
#include "mainglwidget.h"

LoadingScene::LoadingScene()
{
    this->font = new QFont("Arial", 40);
}

void LoadingScene::render()
{
    glColor3d(1, 1, 1);
    QString title = "Loading...";

    QFontMetrics fontMetrics(*this->font);
    int width = (MainGLWidget::getInstance()->width()/2) - fontMetrics.width(title) / 2;
    int height = (MainGLWidget::getInstance()->height()/2) + fontMetrics.height();

    MainGLWidget::getInstance()->renderText(width, height, title, *this->font);
}
