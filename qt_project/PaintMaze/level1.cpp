#include "level1.h"
#include <QVector3D>
#include "wall.h"
#include "endobject.h"

Level1::Level1()
{
    this->timeLimitMillis = 3 * 60 * 1000; // 1 minute
    fillListFormNonCollidable();
    fillListFormCollidable();
}

void Level1::fillListFormNonCollidable()
{

}

void Level1::fillListFormCollidable()
{
    double xScale = 4.0;
    double yScale = 1.0;
    double zScale = -4.0;

    this->startingPoint = new QVector3D(10, 1.85, 10);
//    this->startingPoint = new QVector3D(-5 + xScale*27, 1.85, -1-zScale*27);

    this->goalPoint = new QVector3D(xScale*27, yScale*0, -zScale*28-1);

    this->endObject = new EndObject(this->goalPoint);

        // Wall 1
    QVector3D p11(4*xScale, 0*yScale, -4*zScale);
    QVector3D p12(6*xScale, 0*yScale, -4*zScale);
    QVector3D p13(6*xScale, 0*yScale, -30*zScale);
    QVector3D p14(4*xScale, 0*yScale, -30*zScale);

    this->listShapeCollidable.append(new Wall(&p11, &p12, &p13, &p14));

    // Wall 2
    QVector3D p21(6*xScale, 0*yScale, -14*zScale);
    QVector3D p22(14*xScale, 0*yScale, -14*zScale);
    QVector3D p23(14*xScale, 0*yScale, -16*zScale);
    QVector3D p24(6*xScale, 0*yScale, -16*zScale);

    this->listShapeCollidable.append(new Wall(&p21, &p22, &p23, &p24));

    // Wall 3
    QVector3D p31(0*xScale, 0*yScale, -22*zScale);
    QVector3D p32(10*xScale, 0*yScale, -22*zScale);
    QVector3D p33(10*xScale, 0*yScale, -24*zScale);
    QVector3D p34(0*xScale, 0*yScale, -24*zScale);

    this->listShapeCollidable.append(new Wall(&p31, &p32, &p33, &p34));

    // Wall 4
    QVector3D p41(10*xScale, 0*yScale, -18*zScale);
    QVector3D p42(18*xScale, 0*yScale, -18*zScale);
    QVector3D p43(18*xScale, 0*yScale, -20*zScale);
    QVector3D p44(10*xScale, 0*yScale, -20*zScale);

    this->listShapeCollidable.append(new Wall(&p41, &p42, &p43, &p44));

    // Wall 5
    QVector3D p51(10*xScale, 0*yScale, -20*zScale);
    QVector3D p52(12*xScale, 0*yScale, -20*zScale);
    QVector3D p53(12*xScale, 0*yScale, -28*zScale);
    QVector3D p54(10*xScale, 0*yScale, -28*zScale);

    this->listShapeCollidable.append(new Wall(&p51, &p52, &p53, &p54));

    // Wall 6
    QVector3D p61(8*xScale, 0*yScale, -6*zScale);
    QVector3D p62(18*xScale, 0*yScale, -6*zScale);
    QVector3D p63(18*xScale, 0*yScale, -8*zScale);
    QVector3D p64(8*xScale, 0*yScale, -8*zScale);

    this->listShapeCollidable.append(new Wall(&p61, &p62, &p63, &p64));

    // Wall 7
    QVector3D p71(12*xScale, 0*yScale, -2*zScale);
    QVector3D p72(14*xScale, 0*yScale, -2*zScale);
    QVector3D p73(14*xScale, 0*yScale, -6*zScale);
    QVector3D p74(12*xScale, 0*yScale, -6*zScale);

    this->listShapeCollidable.append(new Wall(&p71, &p72, &p73, &p74));

    // Wall 8
    QVector3D p81(18*xScale, 0*yScale, -6*zScale);
    QVector3D p82(20*xScale, 0*yScale, -6*zScale);
    QVector3D p83(20*xScale, 0*yScale, -14*zScale);
    QVector3D p84(18*xScale, 0*yScale, -14*zScale);

    this->listShapeCollidable.append(new Wall(&p81, &p82, &p83, &p84));

    // Wall 9
    QVector3D p91(20*xScale, 0*yScale, -10*zScale);
    QVector3D p92(32*xScale, 0*yScale, -10*zScale);
    QVector3D p93(32*xScale, 0*yScale, -12*zScale);
    QVector3D p94(20*xScale, 0*yScale, -12*zScale);

    this->listShapeCollidable.append(new Wall(&p91, &p92, &p93, &p94));

    // Wall 10
    QVector3D p101(24*xScale, 0*yScale, 0*zScale);
    QVector3D p102(26*xScale, 0*yScale, 0*zScale);
    QVector3D p103(26*xScale, 0*yScale, -8*zScale);
    QVector3D p104(24*xScale, 0*yScale, -8*zScale);

    this->listShapeCollidable.append(new Wall(&p101, &p102, &p103, &p104));

    // Wall 11
    QVector3D p111(28*xScale, 0*yScale, -2*zScale);
    QVector3D p112(30*xScale, 0*yScale, -2*zScale);
    QVector3D p113(30*xScale, 0*yScale, -10*zScale);
    QVector3D p114(28*xScale, 0*yScale, -10*zScale);

    this->listShapeCollidable.append(new Wall(&p111, &p112, &p113, &p114));

    // Wall 12
    QVector3D p121(18*xScale, 0*yScale, -16*zScale);
    QVector3D p122(20*xScale, 0*yScale, -16*zScale);
    QVector3D p123(20*xScale, 0*yScale, -22*zScale);
    QVector3D p124(18*xScale, 0*yScale, -22*zScale);

    this->listShapeCollidable.append(new Wall(&p121, &p122, &p123, &p124));

    // Wall 13
    QVector3D p131(18*xScale, 0*yScale, -24*zScale);
    QVector3D p132(32*xScale, 0*yScale, -24*zScale);
    QVector3D p133(32*xScale, 0*yScale, -26*zScale);
    QVector3D p134(18*xScale, 0*yScale, -26*zScale);

    this->listShapeCollidable.append(new Wall(&p131, &p132, &p133, &p134));

    // Wall 14
    QVector3D p141(24*xScale, 0*yScale, -18*zScale);
    QVector3D p142(26*xScale, 0*yScale, -18*zScale);
    QVector3D p143(26*xScale, 0*yScale, -20*zScale);
    QVector3D p144(24*xScale, 0*yScale, -20*zScale);

    this->listShapeCollidable.append(new Wall(&p141, &p142, &p143, &p144));

    // Wall 15
    QVector3D p151(24*xScale, 0*yScale, -16*zScale);
    QVector3D p152(30*xScale, 0*yScale, -16*zScale);
    QVector3D p153(30*xScale, 0*yScale, -18*zScale);
    QVector3D p154(24*xScale, 0*yScale, -18*zScale);

    this->listShapeCollidable.append(new Wall(&p151, &p152, &p153, &p154));

    // Wall 16
    QVector3D p161(30*xScale, 0*yScale, -18*zScale);
    QVector3D p162(32*xScale, 0*yScale, -18*zScale);
    QVector3D p163(32*xScale, 0*yScale, -24*zScale);
    QVector3D p164(30*xScale, 0*yScale, -24*zScale);

    this->listShapeCollidable.append(new Wall(&p161, &p162, &p163, &p164));

    // Border Wall 1
    QVector3D pb11(-2*xScale, 0*yScale, 0*zScale);
    QVector3D pb12(0*xScale, 0*yScale, 0*zScale);
    QVector3D pb13(0*xScale, 0*yScale, -30*zScale);
    QVector3D pb14(-2*xScale, 0*yScale, -30*zScale);

    this->listShapeCollidable.append(new Wall(&pb11, &pb12, &pb13, &pb14));

    // Border Wall 2
    QVector3D pb21(0*xScale, 0*yScale, -30*zScale);
    QVector3D pb22(32*xScale, 0*yScale, -30*zScale);
    QVector3D pb23(32*xScale, 0*yScale, -32*zScale);
    QVector3D pb24(0*xScale, 0*yScale, -32*zScale);

    this->listShapeCollidable.append(new Wall(&pb21, &pb22, &pb23, &pb24));

    // Border Wall 3
    QVector3D pb31(32*xScale, 0*yScale, 0*zScale);
    QVector3D pb32(34*xScale, 0*yScale, 0*zScale);
    QVector3D pb33(34*xScale, 0*yScale, -30*zScale);
    QVector3D pb34(32*xScale, 0*yScale, -30*zScale);

    this->listShapeCollidable.append(new Wall(&pb31, &pb32, &pb33, &pb34));

    // Border Wall 4
    QVector3D pb41(0*xScale, 0*yScale, 2*zScale);
    QVector3D pb42(32*xScale, 0*yScale, 2*zScale);
    QVector3D pb43(32*xScale, 0*yScale, 0*zScale);
    QVector3D pb44(0*xScale, 0*yScale, 0*zScale);

    this->listShapeCollidable.append(new Wall(&pb41, &pb42, &pb43, &pb44));
}
