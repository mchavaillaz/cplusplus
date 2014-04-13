#include <QGLWidget>
#include <QDebug>
#include "mainglwidget.h"
#include "ground.h"
#include "face.h"
#include "stain.h"
#include "ball.h"

double Ground::edge = 9000;

Ground::Ground() : Collidable_A(new QVector3D(-edge/2.0, 0, -edge/2.0))
{
    this->p1 = new QVector3D(edge/2.0, 0, -edge/2.0);
    this->p2 = new QVector3D(edge/2.0, 0, edge/2.0);
    this->p3 = new QVector3D(-edge/2.0, 0, edge/2.0);

    // charge la texture depuis un fichier
    QImage image;
    image.load("res/tex/grass.jpg");
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
}


void Ground::draw() {
    glEnable(GL_TEXTURE_2D);

    double textureEdge = edge;

    // sol
    glBindTexture (GL_TEXTURE_2D, texID);

    //glColor3d(1, 1, 1);
    glColor3f(1, 1, 1);
    glBegin(GL_QUADS);
        glTexCoord2d(0, 0);
        glVertex3d(-edge/2.0, 0, -edge/2.0);
        glTexCoord2d(textureEdge, 0);
        glVertex3d(edge/2.0, 0, -edge/2.0);
        glTexCoord2d(textureEdge, textureEdge);
        glVertex3d(edge/2.0, 0, edge/2.0);
        glTexCoord2d(0, textureEdge);
        glVertex3d(-edge/2.0, 0, edge/2.0);
    glEnd();
    glDisable(GL_TEXTURE_2D);
}

Ground::~Ground()
{
    try
    {
        delete this->p1;
        delete this->p2;
        delete this->p3;
    }
    catch(...)
    {
        qDebug() << "destruction de ground erreur";
    }
}

bool Ground::isColliding(Collidable_A *other)
    {
    return other->getP()->y() <= 0;
    }

void Ground::createStain(QVector3D *p, QVector3D* pOrigine, Ball *ball, double angle)
    {
    Face *face = new Face();
    face->add(this->p);
    face->add(p1);
    face->add(p2);
    face->add(p3);
    double tmp = 0;
    face->calculDistance(p, &tmp);
    Stain *tache = new Stain(face, p, pOrigine, new QColor(ball->getColor()));
    emit signal_addTache(tache);
    }
