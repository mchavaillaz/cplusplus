#include <QGLWidget>
#include "decor.h"
#include "point3d.h"
#include "angle3d.h"

Decor::Decor(QImage &image, Point3D &p1, Point3D &p2, Angle3D &angle) : Shape_A(p1)
{
    this->p1 = p1;
    this->p2 = p2;
    this->angle = angle;

    image = QGLWidget::convertToGLFormat(image);

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


void Decor::draw()
{
    glEnable(GL_TEXTURE_2D);
    glBindTexture (GL_TEXTURE_2D, texID);

    glColor3d(1, 1, 1); // blanc
    glBegin(GL_QUADS);
    glTexCoord2d(0, 0);
    glVertex3d(p1.getX(), p1.getY(), p1.getZ());
    glTexCoord2d(100, 0);
    glVertex3d(p2.getX(), p1.getY(), p1.getZ());
    glTexCoord2d(100, 100);
    glVertex3d(p2.getX(), p2.getY(), p2.getZ());
    glTexCoord2d(0, 100);
    glVertex3d(p1.getX(), p2.getY(), p2.getZ());
    glEnd();

    glDisable(GL_TEXTURE_2D);
}
