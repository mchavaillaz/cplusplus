#include <QtGui/QApplication>
#include "mainglwidget.h"
#include "levelscene.h"

int main(int argc, char *argv[])
{
    QApplication a(argc, argv);

    MainGLWidget* widget = MainGLWidget::getInstance();
    widget->init();
    widget->show();
    return a.exec();
}
