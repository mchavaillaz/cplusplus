#include <QGLWidget>
#include <QDebug>
#include "mainglwidget.h"
#include "landscape.h"

double Landscape::edge1 = 9000;

Landscape::Landscape() : Shape_A(new QVector3D(-edge1/2.0, 0, -edge1/2.0))
{
    this->p1 = new QVector3D(edge1/2.0, 0, -edge1/2.0);
    this->p2 = new QVector3D(edge1/2.0, 0, edge1/2.0);
    this->p3 = new QVector3D(-edge1/2.0, 0, edge1/2.0);

    // charge la texture depuis un fichier
    QImage image;
    image.load("res/tex/SuperMario.png");
    image = QGLWidget::convertToGLFormat(image);

    this->image = image;

    glGenTextures(1, &texID);
    glBindTexture(GL_TEXTURE_2D, texID);
    glPixelStorei(GL_UNPACK_ALIGNMENT, 1);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT);
    glTexParameteri (GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT);
    glTexParameteri (GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
    glTexParameteri (GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    glTexEnvf(GL_TEXTURE_ENV, GL_TEXTURE_ENV_MODE, GL_MODULATE);
    glTexImage2D (GL_TEXTURE_2D, 0, GL_RGB, image.width(), image.height(), 0, GL_RGBA , GL_UNSIGNED_BYTE, image.bits());

    QImage image1;
    image1.load("res/tex/sky.png");
    image1 = QGLWidget::convertToGLFormat(image1);

    this->image1 = image1;

    glGenTextures(1, &texID1);
    glBindTexture(GL_TEXTURE_2D, texID1);
    glPixelStorei(GL_UNPACK_ALIGNMENT, 1);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT);
    glTexParameteri (GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT);
    glTexParameteri (GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
    glTexParameteri (GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    glTexEnvf(GL_TEXTURE_ENV, GL_TEXTURE_ENV_MODE, GL_MODULATE);
    glTexImage2D (GL_TEXTURE_2D, 0, GL_RGB, image1.width(), image1.height(), 0, GL_RGBA , GL_UNSIGNED_BYTE, image1.bits());
}

void Landscape::draw() {
    glEnable(GL_TEXTURE_2D);

    int edgelol = 1;
    int edge = edge1/3;

    // sol
    glBindTexture (GL_TEXTURE_2D, texID);

    glColor3d(1, 1, 1);

    glBegin(GL_QUADS);

        // Face1
        glTexCoord2d(0, 0);
        glVertex3d(this->p->x(), this->p->y(), this->p->z());
        glTexCoord2d(edgelol, 0);
        glVertex3d(this->p1->x(), this->p1->y(), this->p1->z());
        glTexCoord2d(edgelol, edgelol);
        glVertex3d(this->p1->x(), edge, this->p1->z());
        glTexCoord2d(0, edgelol);
        glVertex3d(this->p->x(), edge, this->p->z());

        // Face2
        glTexCoord2d(0, 0);
        glVertex3d(this->p1->x(), this->p1->y(), this->p1->z());
        glTexCoord2d(edgelol, 0);
        glVertex3d(this->p2->x(), this->p2->y(), this->p2->z());
        glTexCoord2d(edgelol, edgelol);
        glVertex3d(this->p2->x(), edge, this->p2->z());
        glTexCoord2d(0, edgelol);
        glVertex3d(this->p1->x(), edge, this->p1->z());

        // Face3
        glTexCoord2d(0, 0);
        glVertex3d(this->p2->x(), this->p2->y(), this->p2->z());
        glTexCoord2d(edgelol, 0);
        glVertex3d(this->p3->x(), this->p3->y(), this->p3->z());
        glTexCoord2d(edgelol, edgelol);
        glVertex3d(this->p3->x(), edge, this->p3->z());
        glTexCoord2d(0, edgelol);
        glVertex3d(this->p2->x(), edge, this->p2->z());

        // Face4
        glTexCoord2d(0, 0);
        glVertex3d(this->p3->x(), this->p3->y(), this->p3->z());
        glTexCoord2d(edgelol, 0);
        glVertex3d(this->p->x(), this->p->y(), this->p->z());
        glTexCoord2d(edgelol, edgelol);
        glVertex3d(this->p->x(), edge, this->p->z());
        glTexCoord2d(0, edgelol);
        glVertex3d(this->p3->x(), edge, this->p3->z());

    glEnd();

    glBindTexture (GL_TEXTURE_2D, texID1);

    glColor3d(1, 1, 1);

    glBegin(GL_QUADS);
        glTexCoord2d(0, 0);
        glVertex3d(this->p->x(), edge, this->p->z());
        glTexCoord2d(edgelol, 0);
        glVertex3d(this->p1->x(), edge, this->p1->z());
        glTexCoord2d(edgelol, edgelol);
        glVertex3d(this->p2->x(), edge, this->p2->z());
        glTexCoord2d(0, edgelol);
        glVertex3d(this->p3->x(), edge, this->p3->z());

    glEnd();
}

Landscape::~Landscape()
{
    try
    {
    delete this->p1;
    delete this->p2;
    delete this->p3;
    }
    catch(...)
    {
        qDebug() << "destruction de landscape erreur";
    }
}
