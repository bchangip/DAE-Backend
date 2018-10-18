# -*- coding: utf-8 -*-
"""
Created on Tue Sep 26 01:18:38 2017

@author: ANGELFRANCISCOMORALE
"""

from Tkinter import *
import csv, thread, time, sys, gevent
import pyglet
import ExcelReader
from msvcrt import getch

pyglet.lib.load_library('avbin')
pyglet.have_avbin=True

sys.path.insert(0, 'C:/Users/ANGELFRANCISCOMORALE/OneDrive/Documents/Semestres/10mo/MP/Emotrix/emokit')

from emotiv import Emotiv

class AudioPresentation():
    p = 0
    sounds = []
    index = 0
    time = 3000
    filename='data.csv'
    #//time_block = 5 #tiempo que dura cada estimulo (intervalos)
    sensors = ['F3','F4','AF3','AF4','O1','O2']
    content = []
    def __init__(self):
        with open('order_first_run.txt') as f:
            self.content = f.readlines()
        self.content = [x.strip() for x in self.content]
        content2 = []
        for line in self.content:
            line = line.split(',', 1)
            content2.append(line)
        self.content = content2
        self.player = pyglet.media.Player()
        self.emotions = []
        self.selectedEmotion=''
        for name in self.content:
            soundFile = name[0]
            #f = open(soundFile)
            #print soundFile
            song = pyglet.media.StaticSource(pyglet.media.load(soundFile))
            self.player.queue(song)
            #self.sounds.append(soundFile)
            self.emotions.append(name[1].split(',')[0])
            #extension = self.getAudioExtension(soundFile)

        ########EMOTIV###########
    def getAudioExtension(self,audioName):
        extension = audioName.split('.')[1]
        return extension   
    def setEmotion(self,emotion):
        print emotion
        self.selectedEmotion = emotion
    def setFileName(self,name):
        self.filename = 'C:/Users/ANGELFRANCISCOMORALE/Desktop/data/'+name+'-data.csv'
    def startThreads(self):
        thread.start_new_thread(self.playerMusic,())
        thread.start_new_thread(self.emotiv, ())
        #thread.start_new_thread(self.selectEmotion, ())
    def emotiv(self):
        headset = Emotiv()
        gevent.spawn(headset.setup)
        print("Serial Number: %s" % headset.serial_number)

        self.num_blocks = len(self.content)
        cont_seconds = 0
        temp_t = 0
        tag = self.content[0][1]
        #Se define el escritor de las lecturas en el archivo CSV
        writer = csv.writer(open(self.filename, 'w'), delimiter=',', quotechar='"', lineterminator='\n')
        row = ["Audio"]
        row.append("Time")
        row.append("Exact Time")
        for sensor in self.sensors:
            row.append(sensor)
            row.append(sensor + "_Quality")
        row.append("Emotion")
        row.append("Selected Emotion")
        writer.writerow(row)
        try:
            t0 = time.time()
            time_block = 300
            index_actual = 0
            while True:
                t = int(time.time()-t0)
                if temp_t != t:
                    cont_seconds += 1
                if cont_seconds >= time_block:
                    cont_seconds = 0
                    index_actual+=1 
                    #self.player.next_source()
                if cont_seconds >= 300:
                    time_block = 15
                if t > 720:
                    headset.close()
                    self.player.pause()
                    break
                if t < 180:
                    tag = 'NON-RELAX'
                elif (t >= 180):
                    tag = self.emotions[index_actual]
                row = [self.content[index_actual][0].split('.')[0]]
                row.append(str(t))
                row.append(time.time())
                try:
                    for sensor in self.sensors:
                        row.append(str(headset.sensors[sensor]['value']))
                        row.append(str(headset.sensors[sensor]['quality']))
                except:
                    print "Sensor incorrecto"
                    headset.close()
                row.append(tag)
                row.append(self.selectedEmotion)
                if self.selectedEmotion != '':
                    self.selectedEmotion = ''
                # Se exporta a csv
                writer.writerow(row)
                #print row
                temp_t = t
                gevent.sleep(0)
        except KeyboardInterrupt:
            headset.close()
        finally:
            headset.close()
        ####################
    #def startThreads(self):
    def playerMusic(self):
        self.player.play()
        
        #self.VisualGUI = Button(self.root,text="Estimulos Visuales",command=self.visualGUI)
        #self.VisualGUI.grid(row=8,column=0)     
"""    def visualGUI(self):
        app = ImagePresentation()
    def auralGUI(self):
        app = AudioPresentation()"""

def createWidgets():
    root.matrix = ExcelReader.getMatrix()
    root.studentNamesLabel = Label(root,text='Nombre estudiante: ',width=40)
    root.studentNamesLabel.grid(row=0,column=0)
    root.studentNames = Spinbox(root,values=getColumnValues(root,'nombre'), width=40, state="readonly", wrap=True)
    root.studentNames.configure(command=updateProfile)
    root.studentNames.grid(row=0,column=1)
    root.IDLabel = Label(root,text='ID',width=50)
    root.IDLabel.grid(row=1,column=0)
    root.ID = Entry(root, width=50)
    root.ID.grid(row=1,column=1)
    root.CodigoLabel = Label(root,text='Código',width=50)
    root.CodigoLabel.grid(row=2,column=0)
    root.Codigo = Entry(root, width=50)
    root.Codigo.grid(row=2,column=1)
    root.CarneLabel = Label(root,text='Carné',width=50)
    root.CarneLabel.grid(row=3,column=0)
    root.Carne = Entry(root, width=50)
    root.Carne.grid(row=3,column=1)
    root.CorreoLabel = Label(root,text='Correo',width=50)
    root.CorreoLabel.grid(row=4,column=0)
    root.Correo = Entry(root, width=50)
    root.Correo.grid(row=4,column=1)
    root.CarreraLabel = Label(root,text='Año y carrera',width=50)
    root.CarreraLabel.grid(row=5,column=0)
    root.Carrera = Entry(root, width=50)
    root.Carrera.grid(row=5,column=1)
    root.TelefonoLabel = Label(root,text='Teléfono',width=50)
    root.TelefonoLabel.grid(row=6,column=0)
    root.Telefono = Entry(root, width=50)
    root.Telefono.grid(row=6,column=1)
    root.AuralGUI = Button(root,text="Estimulos Auditivos",command=createApp)
    root.AuralGUI.grid(row=8,column=0) 

def createApp():
    app.setFileName(root.Codigo.get())
    app.startThreads()
    
def key(event):
    letra = event.char
    #print letra
    if (letra in "aaa"):
        app.setEmotion('happy')
    elif (letra in "sss"):
        app.setEmotion('sad')
    else:
        app.setEmotion('other')

def getColumnValues(root,colName):
    columnas ={
        'numero': 0,
        'codigo': 1,
        'nombre': 2,
        'carne': 3,
        'correo': 4,
        'anio': 5,
        'telefono': 6,
        'fecha': 7,
        'hora' : 8 
    }
    values = []
    for i in root.matrix[1:len(root.matrix)]:
        if i[columnas[colName]] is None:
            values.append('')
        else:
            values.append(i[columnas[colName]])
    return values 

def updateProfile():
    currentName = root.studentNames.get()
    #GetIndex
    names = getColumnValues(root,'nombre')
    i = names.index(currentName)+1
    ID = root.matrix[i][0]
    codigo = root.matrix[i][1]
    carne = root.matrix[i][3]
    correo = root.matrix[i][4]
    carrera = root.matrix[i][5]
    telefono = root.matrix[i][6]
    root.ID.delete(0,END)
    root.ID.insert(0,ID)
    root.Codigo.delete(0,END)
    root.Codigo.insert(0,codigo)
    root.Carne.delete(0,END)
    root.Carne.insert(0,carne)
    root.Correo.delete(0,END)
    root.Correo.insert(0,correo)
    root.Carrera.delete(0,END)
    root.Carrera.insert(0,carrera)
    root.Telefono.delete(0,END)
    root.Telefono.insert(0,telefono)   

root = Tk()
root.title('Emotrix 2017 Estimulación Auditiva')
createWidgets();
app = AudioPresentation()
root.bind("<Key>",key)
w = root.winfo_screenwidth()
h = root.winfo_screenheight()
root.geometry("%dx%d+%d+%d" % (w,h, 0, 0))
root.mainloop()
