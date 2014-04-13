/********************************************************************************
** Form generated from reading UI file 'paintmazewindow.ui'
**
** Created: Tue 15. Jan 14:14:03 2013
**      by: Qt User Interface Compiler version 4.7.3
**
** WARNING! All changes made in this file will be lost when recompiling UI file!
********************************************************************************/

#ifndef UI_PAINTMAZEWINDOW_H
#define UI_PAINTMAZEWINDOW_H

#include <QtCore/QVariant>
#include <QtGui/QAction>
#include <QtGui/QApplication>
#include <QtGui/QButtonGroup>
#include <QtGui/QHeaderView>
#include <QtGui/QMainWindow>
#include <QtGui/QWidget>

QT_BEGIN_NAMESPACE

class Ui_PaintMazeWindow
{
public:
    QWidget *centralWidget;

    void setupUi(QMainWindow *PaintMazeWindow)
    {
        if (PaintMazeWindow->objectName().isEmpty())
            PaintMazeWindow->setObjectName(QString::fromUtf8("PaintMazeWindow"));
        PaintMazeWindow->resize(400, 300);
        centralWidget = new QWidget(PaintMazeWindow);
        centralWidget->setObjectName(QString::fromUtf8("centralWidget"));
        PaintMazeWindow->setCentralWidget(centralWidget);

        retranslateUi(PaintMazeWindow);

        QMetaObject::connectSlotsByName(PaintMazeWindow);
    } // setupUi

    void retranslateUi(QMainWindow *PaintMazeWindow)
    {
        PaintMazeWindow->setWindowTitle(QApplication::translate("PaintMazeWindow", "PaintMazeWindow", 0, QApplication::UnicodeUTF8));
    } // retranslateUi

};

namespace Ui {
    class PaintMazeWindow: public Ui_PaintMazeWindow {};
} // namespace Ui

QT_END_NAMESPACE

#endif // UI_PAINTMAZEWINDOW_H
