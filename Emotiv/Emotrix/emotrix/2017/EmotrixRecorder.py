# -*- coding: utf-8 -*-
# **********************************************************************************************************************
# Archivo:      EmotrixRecorder.py
# Proposito:    Este archivo es el encargado de leer la data directamente desde el EMOTIV, y guardarla en un archivo csv
#               para luego hacer un preprocesamiento de ella.
# Autor:        Diego Jacobs    13160
# **********************************************************************************************************************

import platform
import csv
import time
import sys
sys.path.insert(0, '/home/emotrix/Documents/EMOTRIX/emokit')

from emotiv import Emotiv
import gevent


class EmotrixRecoder(object):

    #El metodo init define algunos parametros por defecto, para almacenar las lecturas del EMOTIV
    def __init__(self, fileName):
        self.sequence = ['happy', 'neutral', 'sad', 'happy', 'neutral', 'sad', 'happy', 'neutral', 'sad', 'happy', 'neutral', 'sad','happy', 'neutral', 'sad','happy', 'neutral', 'sad','happy', 'neutral', 'sad']
        self.time_block = 7
        self.num_blocks = len(self.sequence)
        self.filename = fileName

    #Metodo para iniciar la grabacion de las lecturas de las señales EEG.
    def start(self, sequence=None, sensors=None, time_block=7):
        self.time_block = time_block
        #Se define el objeto EMOTIV, utilizando la libreria EMOKIT
        headset = Emotiv()
        gevent.spawn(headset.setup)
        
        print("Serial Number: %s" % headset.serial_number)

        if sequence is not None:
            self.sequence = sequence
            self.num_blocks = len(self.sequence)
        i = 0
        cont_block = 0
        cont_seconds = 0
        temp_t = 0
        tag = self.sequence[0]

        #Se define el escritor de las lecturas en el archivo CSV
        # writer = csv.writer(open(self.filename, 'w'), delimiter=',', quotechar='"', lineterminator='\n') #windows
        writer = csv.writer(open(self.filename, 'w'), delimiter=',', quotechar='"') #linux
        try:
            row = ["Time"]
            for sensor in sensors:
                row.append(sensor)
                row.append(sensor + "_Quality")
            row.append("Tag")

            writer.writerow(row)

            t0 = time.time()
            while True:
                t = int(time.time()-t0)
                
                if temp_t != t:
                    cont_seconds += 1

                if cont_seconds > self.time_block:
                    cont_seconds = 0
                    cont_block += 1
                    if cont_block == self.num_blocks:
                        headset.close()
                        break
                    else:
                        tag = self.sequence[cont_block]

                # Se construye la informacion a guardar
                row = [str(t)]
                try:
                    for sensor in sensors:
                        row.append(str(headset.sensors[sensor]['value']))
                        row.append(str(headset.sensors[sensor]['quality']))
                    
                except:
                    print "Sensor incorrecto"
                    headset.close()
                row.append(tag)
                
                # Se exporta a csv
                writer.writerow(row)
                print row
                temp_t = t
                gevent.sleep(0)
        except KeyboardInterrupt:
            headset.close()
        finally:
            headset.close()
        i += 1


#Se inicia el proceso de la grabacion, se define la secuencia y el tiempo de cada estimulo.
er = EmotrixRecoder("data.csv")
#Secuencia de imagenes, Sensores a leer, tiempo de corrida
er.start(['NOPE','RELAX', 'RELAX', 'RELAX', 'RELAX',
          'HAPPY', 'NEUTRAL', 'SAD', 'NEUTRAL',
          'HAPPY', 'NEUTRAL', 'SAD', 'NEUTRAL',
          'HAPPY', 'NEUTRAL', 'SAD', 'NEUTRAL',
          'HAPPY', 'NEUTRAL', 'SAD', 'NEUTRAL',
          'HAPPY', 'NEUTRAL', 'SAD', 'NEUTRAL',
          'HAPPY', 'NEUTRAL', 'SAD', 'NEUTRAL',
          'HAPPY', 'NEUTRAL', 'SAD', 'NEUTRAL',
          'HAPPY', 'NEUTRAL', 'SAD', 'NEUTRAL',
          'HAPPY', 'NEUTRAL', 'SAD', 'NEUTRAL',
          'HAPPY', 'NEUTRAL', 'SAD', 'NEUTRAL',
          ], ['F3', 'F4', 'AF3', 'AF4', 'O1', 'O2'], 4)
