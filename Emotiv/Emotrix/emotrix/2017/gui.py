# -*- coding: utf-8 -*-
"""
Created on Tue Sep 19 19:00:52 2017

@author: ANGELFRANCISCOMORALE
"""

from Tkinter import *
from Estimulos_Visuales.main import *
from auditivos.gui import *

class Application(Frame):


    def createWidgets(self):
        self.matrix = ExcelReader.getMatrix()
        self.studentNamesLabel = Label(self,text='Nombre estudiante: ',width=40)
        self.studentNamesLabel.grid(row=0,column=0)
        self.studentNames = Spinbox(self,values=self.getColumnValues('nombre'), width=40, state="readonly", wrap=True, command=self.updateProfile)
        self.studentNames.grid(row=0,column=1)
        self.IDLabel = Label(self,text='ID',width=50)
        self.IDLabel.grid(row=1,column=0)
        self.ID = Entry(self, width=50)
        self.ID.grid(row=1,column=1)
        self.CodigoLabel = Label(self,text='Código',width=50)
        self.CodigoLabel.grid(row=2,column=0)
        self.Codigo = Entry(self, width=50)
        self.Codigo.grid(row=2,column=1)
        self.CarneLabel = Label(self,text='Carné',width=50)
        self.CarneLabel.grid(row=3,column=0)
        self.Carne = Entry(self, width=50)
        self.Carne.grid(row=3,column=1)
        self.CorreoLabel = Label(self,text='Correo',width=50)
        self.CorreoLabel.grid(row=4,column=0)
        self.Correo = Entry(self, width=50)
        self.Correo.grid(row=4,column=1)
        self.CarreraLabel = Label(self,text='Año y carrera',width=50)
        self.CarreraLabel.grid(row=5,column=0)
        self.Carrera = Entry(self, width=50)
        self.Carrera.grid(row=5,column=1)
        self.TelefonoLabel = Label(self,text='Teléfono',width=50)
        self.TelefonoLabel.grid(row=6,column=0)
        self.Telefono = Entry(self, width=50)
        self.Telefono.grid(row=6,column=1)
        
        self.VisualGUI = Button(self,text="Estimulos Visuales",command=self.visualGUI)
        self.VisualGUI.grid(row=8,column=0)
        self.AuralGUI = Button(self,text="Estimulos Auditivos",command=self.auralGUI)
        self.AuralGUI.grid(row=8,column=1)
        
"""    def visualGUI(self):
        app = ImagePresentation()"""
    def auralGUI(self):
        app = AudioPresentation()
        
    def __init__(self, master=None):
        Frame.__init__(self, master)
        self.pack()
        self.createWidgets()
        
    def getColumnValues(self,colName):
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
        for i in self.matrix[1:len(self.matrix)]:
            if i[columnas[colName]] is None:
                values.append('')
            else:
                values.append(i[columnas[colName]])
        return values
    
    
    def updateProfile(self):
        currentName = self.studentNames.get()
        #GetIndex
        names = self.getColumnValues('nombre')
        i = names.index(currentName)+1
        ID = self.matrix[i][0]
        codigo = self.matrix[i][1]
        carne = self.matrix[i][3]
        correo = self.matrix[i][4]
        carrera = self.matrix[i][5]
        telefono = self.matrix[i][6]
        self.ID.delete(0,END)
        self.ID.insert(0,ID)
        self.Codigo.delete(0,END)
        self.Codigo.insert(0,codigo)
        self.Carne.delete(0,END)
        self.Carne.insert(0,carne)
        self.Correo.delete(0,END)
        self.Correo.insert(0,correo)
        self.Carrera.delete(0,END)
        self.Carrera.insert(0,carrera)
        self.Telefono.delete(0,END)
        self.Telefono.insert(0,telefono)
        
        

root = Tk()
root.title('Megaproyecto Emotrix 2017')
app = Application(master=root)
app.mainloop()
root.destroy()