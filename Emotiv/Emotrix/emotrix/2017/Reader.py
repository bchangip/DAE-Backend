# -*- coding: utf-8 -*-
# **********************************************************************************************************************
# Archivo:      Reader.py
# Proposito:    Este archivo es el encargado de leer los archivos csv
# Autor:        Diego Jacobs    13160
# **********************************************************************************************************************

import csv
import numpy as np
from BlockData import BlockData

class Reader(object):
    """
    docstring here
        :param object: 
    """
    
    def __init__(self, fileData="data.csv", sensors=None, tagsToIgnore=None):
        """
        docstring here
            :param self: 
            :param fileData: 
            :param tagsToIgnore: 
        """
        self.fileName = fileData
        self.ignoreTags = tagsToIgnore
        self.time = np.array([])
        self.f3 = np.array([])
        self.f4 = np.array([])
        self.af3 = np.array([])
        self.af4 = np.array([])
        self.o1 = np.array([])
        self.o2 = np.array([])
        self.tag = np.array([])
        self.blocks = []

    def readFile(self):
        """
        docstring here
            :param self: 
        """
        try:
            fileToRead = open(self.fileName, "rb")
            reader = csv.reader(fileToRead, delimiter=',', quotechar='"')
            t, f3, f4, af3, af4, o1, o2, tag = [], [], [], [], [], [], [], []
            i = 0
            for row in reader:
                if (i>0):
                    t.append(int(row[0]))
                    f3.append(int(row[1]))
                    f4.append(int(row[3]))
                    af3.append(int(row[5]))
                    af4.append(int(row[7]))
                    o1.append(int(row[9]))
                    o2.append(int(row[11]))
                    tag.append(row[13])
                i = i + 1

            self.time = np.append(self.time, np.array(t))
            self.tag = np.append(self.tag, np.array(tag))
            self.f3 = np.append(self.f3, np.array(f3))
            self.f4 = np.append(self.f4, np.array(f4))
            self.af3 = np.append(self.af3, np.array(af3))
            self.af4 = np.append(self.af4, np.array(af4))
            self.o1 = np.append(self.o1, np.array(o1))
            self.o2 = np.append(self.o2, np.array(o2))
        except:
            print "Archivo no encontrado: "  + self.fileName

    #Metodo para obtener los datos en bloques de 1 segundo.
    def getBlocksData(self):
        self.blocks = []
        before = self.tag[0]
        lower_limit_i = 0
        i = 0
        size = 256
        overlap = 20
        for t in np.append(self.tag, 'fin'):
            if t != before:
                previous_second = self.time[lower_limit_i]
                for j in range(lower_limit_i, i-128, overlap):
                    #Se descarta el primer segundo de cada bloque
                    if j != lower_limit_i:
                        block = BlockData(tag=before, time=self.time[j], f3=self.f3[j:j+size], f4=self.f4[j:j+size], af3=self.af3[j:j+size], af4=self.af4[j:j+size])
                        self.blocks.append(block)
                lower_limit_i = i
                before = t
            i += 1
        return self.blocks

r = Reader("/home/emotrix/Documents/EMOTRIX/emotrix/2017/data.csv", ["F3", "F4", "AF3", "AF4"], ["RELAX", "NOPE"])
r.readFile()
