# coding: utf-8
from OpenGL.GL import *
from OpenGL.GLU import *
from OpenGL.GLUT import *
import random
'''全局参数开始'''
life_down_p = 2 #竞争参数下限
life_up_p = 3 #竞争参数上限
life_die_time = 5 #死亡时间
life_begin = 1000 #开局生成时间
map_size = 100
'''全局参数结束'''
num = 0 #golbal
life_map = [0]*map_size*map_size #golbal
life_new = [0]*map_size*map_size #golbal
all_c = [0]*map_size*map_size
green_c = [0]*map_size*map_size
red_c = [0]*map_size*map_size
w = 2/map_size #width pre
h = 2/map_size #height pre
RED = 1
GREEN = 2
def draw_point(color,p) : #画点
    x = int(p%map_size)
    y = int(p/map_size)
    glColor3f(color[0],color[1],color[2])
    glBegin(GL_QUADS)
    glVertex2f(x*w-1,y*h-1)
    glVertex2f((x+1)*w-1,y*h-1)
    glVertex2f((x+1)*w-1,(y+1)*h-1)
    glVertex2f(x*w-1,(y+1)*h-1)
    glEnd()
def god() :
    global life_map,num,font_map,all_c,green_c,red_c
    if num < life_begin : #初始生成开始
        num += 1
        x = random.randint(1,map_size-2)*map_size+random.randint(1,map_size-2)
        if random.randint(0,1) : #绿色生物
            life_map[x] = GREEN
            draw_point([0,1,0],x)
        else : #红色生物
            life_map[x] = RED
            draw_point([1,0,0],x)
    else : #初始生成结束，开始繁殖
        '''情况判断开始'''
        for x in range(0,map_size) :
            for y in range(0,map_size) :
                i = y*map_size+x
                '''获取周边信息'''
                c = [(y-1)%map_size*map_size+(x-1)%map_size,
                     (y-1)%map_size*map_size+ x            ,
                     (y-1)%map_size*map_size+(x+1)%map_size,
                      y            *map_size+(x-1)%map_size,
                      y            *map_size+(x+1)%map_size,
                     (y+1)%map_size*map_size+(x-1)%map_size,
                     (y+1)%map_size*map_size+ x            ,
                     (y+1)%map_size*map_size+(x+1)%map_size,]
                red_c[i],green_c[i],all_c[i] = 0,0,0
                for cc in c :
                    if life_map[cc] == GREEN :
                        green_c[i] += 1
                    elif life_map[cc] == RED :
                        red_c[i] += 1
                all_c[i] = green_c[i] + red_c[i]
        '''判断'''
        for i in range(0,map_size*map_size) :
            if all_c[i] == life_up_p : #生存
                if green_c[i] > red_c[i] :
                    life_map[i] = GREEN
                    draw_point([0,1,0],i)
                elif green_c[i] < red_c[i] :
                    life_map[i] = RED
                    draw_point([1,0,0],i)
                else :
                    if random.randint(0,1) :
                        life_map[i] = GREEN
                        draw_point([0,1,0],i)
                    else :
                        life_map[i] = RED
                        draw_point([1,0,0],i)
            elif all_c[i] > life_up_p or all_c[i] < life_down_p : #死亡
                life_map[i] = 0
                draw_point([0,0,0],i)
            #else : 保持
def drawFunc() :
    god()
    glFlush()
glutInit()
glutInitDisplayMode(GLUT_SINGLE | GLUT_RGBA)
glutInitWindowSize(800,800)
glutCreateWindow(b"life-forver")
glutDisplayFunc(drawFunc)
glutIdleFunc(drawFunc)
glutMainLoop()
