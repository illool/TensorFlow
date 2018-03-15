from OpenGL.GL import *
from OpenGL.GLUT import *
from OpenGL.GLU import gluLookAt,gluPerspective
from math import cos,sin,pi
from time import sleep
import sys

PI2=pi*2.0
WIN_X=300
WIN_Y=300
ANG=0.0
ANGX=0.0
theVortex=0
winid=0

def vortex(R=20.0,r=12.0):
    ''' Torus_Vortex '''
    nparts=50
    mparts=28
    detail=float(mparts)/float(nparts)
    tm=0.0
    
    for m in range(mparts):
        m=float(m)
        c=float(m%2)
        glColor3f(c*0.5,c*0.8,c*1.0)
        glBegin(GL_QUAD_STRIP)
        move=0.0
        for n in range(nparts+1):
            n=float(n)
            move+=detail
            x=r*cos(n/nparts*PI2)
            y=r*sin(n/nparts*PI2)
            for o in (0.0,1.0):
                tm=o+m+move;
                mx=(x+R)*cos(tm/mparts*PI2)
                mz=(x+R)*sin(tm/mparts*PI2)
                glVertex3f(mx,y,mz)
        glEnd()

def init():
    global theVortex
    glClearColor(0.0,0.0,0.0,0.0)
    glEnable(GL_DEPTH_TEST)
    theVortex=glGenLists(1)
    glNewList(theVortex,GL_COMPILE)
    vortex(18.0,12.0)
    glEndList()
    
def display():
    global theVortex
    glClearColor(0.0,0.0,0.0,0.0)  
    glClear(GL_COLOR_BUFFER_BIT|GL_DEPTH_BUFFER_BIT)
    glPushMatrix()
    glRotatef(ANGX,1.0,0.0,0.0)
    glRotatef(ANG,0.0,-1.0,0.0)
    glCallList(theVortex)
    glPopMatrix()
    glutSwapBuffers()

def idle():
    global ANG
    ANG+=1.0
    sleep(0.01)
    glutPostRedisplay()

def reshape(Width,Height):
    far=30.0
    if (Width==Height):
        glViewport(0,0,Width,Height)
    elif (Width>Height):
        glViewport(0,0,Height,Height)
    else:
        glViewport(0,0,Width,Width)
    
    glMatrixMode(GL_PROJECTION)
    glLoadIdentity()
    #glFrustum(-10.0,10.0,-10.0,10.0,3.0,60.0)
    gluPerspective(80.0,1.0,1.0,80.0)
    glMatrixMode(GL_MODELVIEW)
    glLoadIdentity()
    gluLookAt(0.0,0.0,far, 0.0,0.0,10.0, 0.0,1.0,far)
    
def hitkey(key,mousex,mousey):
    global winid,ANGX
    if (key=='q'):            
        glutDestroyWindow(winid)
        sys.exit()
    elif (key=='a'):
        ANGX+=1.0

def main():
    global WIN_X,WIN_Y,winid
    glutInit(sys.argv)
    glutInitDisplayMode(GLUT_DOUBLE|GLUT_RGBA|GLUT_DEPTH)
    glutInitWindowSize(WIN_X,WIN_Y)
    glutInitWindowPosition(100,100)
    #winid=glutCreateWindow("Vortex") =>
    #ctypes.ArgumentError: argument 1: <class 'TypeError'>: wrong type
    winid=glutCreateWindow(b"Vortex")
    init()
    glutDisplayFunc(display)
    glutIdleFunc(idle)
    glutReshapeFunc(reshape)
    glutKeyboardFunc(hitkey)
    glutMainLoop()

if __name__=="__main__":
    main()
