# -*- coding: utf-8 -*-
# **********************************************************************************************************************
# Archivo:      emotrix_recorder.py
# Proposito:    Este archivo es el encargado de leer la data directamente desde el EMOTIV, y guardarla en un archivo csv
#               para luego hacer un preprocesamiento de ella.
# Autor:        Emotrix 2016-2017
# **********************************************************************************************************************

import platform
import csv
import time
import sys
sys.path.insert(0, '/home/emotrix/Downloads/EMOTRIX/emokit')

from emotiv import Emotiv
import gevent


class EmotrixRecoder(object):

    #El metodo init define algunos parametros por defecto, para almacenar las lecturas del EMOTIV
    def __init__(self):
        #secuencia de emociones
        self.sequence = ['happy', 'neutral', 'sad', 'happy', 'neutral', 'sad', 'happy', 'neutral', 'sad', 'happy', 'neutral', 'sad','happy', 'neutral', 'sad','happy', 'neutral', 'sad','happy', 'neutral', 'sad']
        self.time_block = 7 #tiempo que dura cada estimulo (intervalos)
        self.num_blocks = len(self.sequence)
        self.filename = 'data.csv'

    #Metodo para iniciar la grabacion de las lecturas de las señales EEG.
    def start(self, sequence=None, time_block=7, filename='data.csv'):
        self.time_block = time_block
        self.filename = filename
        #Se define el objeto EMOTIV, utilizando la libreria EMOKIT
        headset = Emotiv()
        gevent.spawn(headset.setup)
        gevent.sleep(0)
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
        writer = csv.writer(open(self.filename, 'w'), delimiter='\t', quotechar='"')
        try:
            t0 = time.time()
            while True:
                t = int(time.time()-t0)
                #t = int(time.time())
                if t > 4:
                  headset.close()
                  break
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

                # Se obtiene el paquete de datos, utilizando EMOKIT
                packet = headset.dequeue()
                #print packet.sensors
                # Se construye la informacion a guardar
                row = [str(t),
                       "AF3:" + str(packet.sensors['AF3']['quality']) + "," + str(packet.sensors['AF3']['value']),
                       "AF4:" + str(packet.sensors['AF4']['quality']) + "," + str(packet.sensors['AF4']['value']),
                       "F7:" + str(packet.sensors['F7']['quality']) + "," + str(packet.sensors['F7']['value']),
                       "F3:" + str(packet.sensors['F3']['quality']) + "," + str(packet.sensors['F3']['value']),
                       "F4:" + str(packet.sensors['F4']['quality']) + "," + str(packet.sensors['F4']['value']),
                       "F8:" + str(packet.sensors['F8']['quality']) + "," + str(packet.sensors['F8']['value']),
                       "FC5:" + str(packet.sensors['FC5']['quality']) + "," + str(packet.sensors['FC5']['value']),
                       "FC6:" + str(packet.sensors['FC6']['quality']) + "," + str(packet.sensors['FC6']['value']),
                       "T7:" + str(packet.sensors['T7']['quality']) + "," + str(packet.sensors['T7']['value']),
                       "T8:" + str(packet.sensors['T8']['quality']) + "," + str(packet.sensors['T8']['value']),
                       "P7:" + str(packet.sensors['P7']['quality']) + "," + str(packet.sensors['P7']['value']),
                       "P8:" + str(packet.sensors['P8']['quality']) + "," + str(packet.sensors['P8']['value']),
                       "O1:" + str(packet.sensors['O1']['quality']) + "," + str(packet.sensors['O1']['value']),
                       "O2:" + str(packet.sensors['O2']['quality']) + "," + str(packet.sensors['O2']['value']),
                       tag]
                # Se exporta a csv
                writer.writerow(row)
                print (row)
                temp_t = t
                gevent.sleep(0)
        except KeyboardInterrupt:
            headset.close()
        finally:
            headset.close()
        i += 1


#Se inicia el proceso de la grabacion, se define la secuencia y el tiempo de cada estimulo.
er = EmotrixRecoder()
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
          ], 2, 'person.csv')
