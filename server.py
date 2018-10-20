from flask import Flask, request
from flask_socketio import SocketIO, emit, send
from flask_cors import CORS
import time
import multiprocessing
import requests
import pyaudio
import wave
import os
from neupy import algorithms, environment
from pymongo import MongoClient
import numpy as np
import sox
import pickle
from olga import return_values
from leonel import prediction
from castro import expression
import cv2
import glob
import xlrd
import csv
import sys
import mysql.connector
import pandas as pd

import sklearn

from sklearn.model_selection import train_test_split
from sklearn import preprocessing, neighbors, svm

def getSensorArray(person, sensor):
    with open(person+'.csv') as csvfile:
        dataInput = csv.reader(csvfile, delimiter=' ', quotechar='|')
        print('Reading CSV', dataInput)
        array = []
        count_array = 0
        current_sec = 0
        current_count = 0
        current_sum = 0
        for row in dataInput:
            line = ''.join(row)
            splitData = line.split('\t')
            #print 'Second of Data ' + str(splitData[0])
            #Using sensors: AF3, AF4, F3 and F4
            for i in range(1, len(splitData)-1):
                splitDots= splitData[i].split(":")
                splitCommas = splitDots[1].split(',')
                if splitDots[0] == sensor :
                    if current_sec == int(splitData[0]) :
                        current_sum = current_sum + int(splitCommas[1])
                        current_count += 1
                    else:
                        average = current_sum / current_count
                        current_sec = int(splitData[0])
                        current_sum = int(splitCommas[1])
                        current_count= 0
                        array.append(average)
                        #af3 = af3 + str(count_af3) + ',' + str(average) + '\n'
                        count_array = count_array+1
    return array

def getSex(sexo):
  if sexo == 'Masculino':
    return 0
  else:
    return 1

def getDemographic(
    edad,
    pebl, 
    dsmt,
    hare,
    ciep,
    cief,
    ciec,
    ciem,
    ciex,
    cies,
    cie
    ):
    demographics = []
    demographics.append(edad)
    demographics.append(pebl)
    demographics.append(dsmt)
    demographics.append(hare)
    demographics.append(ciep)
    demographics.append(cief)
    demographics.append(ciec)
    demographics.append(ciem)
    demographics.append(ciex)
    demographics.append(cies)
    demographics.append(cie)
    return demographics

def insert_BD (
    second,
    sexo,
    edad,
    pebl, 
    dsmt,
    hare,
    ciep,
    cief,
    ciec,
    ciem,
    ciex,
    cies,
    cie
    ):
  con = mysql.connector.connect(
    host="localhost",
    user="root",
    passwd="admin",
    database="megaproyecto"
  )

  cur = con.cursor()
  person_count = 1
  q_count = 1
  var_count=1
  meassure_count = 1
  cur.execute('DELETE FROM `megaproyecto`.`medicion`;')
  con.commit()
  cur.execute('DELETE FROM `megaproyecto`.`pregunta`;')
  con.commit()
  cur.execute('DELETE FROM `megaproyecto`.`sensor`;')
  con.commit()
  cur.execute('DELETE FROM `megaproyecto`.`variable`;')
  con.commit()
  cur.execute('DELETE FROM `megaproyecto`.`variables`;')
  con.commit()
  cur.execute('DELETE FROM `megaproyecto`.`persona`;')
  con.commit()
  cur.execute('DELETE FROM `megaproyecto`.`campo`;')
  con.commit()
  cur.execute('INSERT INTO `megaproyecto`.`campo` VALUES (%s,%s);', (1, 'Sexo'))
  con.commit()
  cur.execute('INSERT INTO `megaproyecto`.`campo` VALUES (%s,%s);', (2, 'Edad'))
  con.commit()
  cur.execute('INSERT INTO `megaproyecto`.`campo` VALUES (%s,%s);', (3, 'Pebl'))
  con.commit()
  cur.execute('INSERT INTO `megaproyecto`.`campo` VALUES (%s,%s);', (4, 'Dsmt'))
  con.commit()
  cur.execute('INSERT INTO `megaproyecto`.`campo` VALUES (%s,%s);', (5, 'Hare'))
  con.commit()
  cur.execute('INSERT INTO `megaproyecto`.`campo` VALUES (%s,%s);', (6, 'Ciep'))
  con.commit()
  cur.execute('INSERT INTO `megaproyecto`.`campo` VALUES (%s,%s);', (7, 'Cief'))
  con.commit()
  cur.execute('INSERT INTO `megaproyecto`.`campo` VALUES (%s,%s);', (8, 'Ciec'))
  con.commit()
  cur.execute('INSERT INTO `megaproyecto`.`campo` VALUES (%s,%s);', (9, 'Ciem'))
  con.commit()
  cur.execute('INSERT INTO `megaproyecto`.`campo` VALUES (%s,%s);', (10, 'Ciex'))
  con.commit()
  cur.execute('INSERT INTO `megaproyecto`.`campo` VALUES (%s,%s);', (11, 'Cies'))
  con.commit()
  cur.execute('INSERT INTO `megaproyecto`.`campo` VALUES (%s,%s);', (12, 'Cie'))
  con.commit()


  cur.execute('INSERT INTO `megaproyecto`.`sensor` (`id`, `sensor`) VALUES (%s,%s);', (1, 'AF3'))
  con.commit()
  cur.execute('INSERT INTO `megaproyecto`.`sensor` (`id`, `sensor`) VALUES (%s,%s);', (2, 'F3'))
  con.commit()
  cur.execute('INSERT INTO `megaproyecto`.`sensor` (`id`, `sensor`) VALUES (%s,%s);', (3, 'AF4'))
  con.commit()
  cur.execute('INSERT INTO `megaproyecto`.`sensor` (`id`, `sensor`) VALUES (%s,%s);', (4, 'F4'))
  con.commit()

  person = 'person'
  af3 = getSensorArray(person, 'AF3')
  print('something')
  f3 = getSensorArray(person, 'F3')
  af4 = getSensorArray(person, 'AF4')
  f4 = getSensorArray(person, 'F4')
  if af3 != 0:
      person_count += 1
      cur.execute('INSERT INTO `megaproyecto`.`persona` (`idpersona`,`codigo`) VALUES (%s, %s);', (int(person_count), str(person)))
      con.commit()

      cur.execute('INSERT INTO `megaproyecto`.`variables` (`id`,`id_persona`) VALUES (%s,%s);', (person_count-1, person_count))
      con.commit()

      sex = getSex(sexo)
      if sex != 2:
          cur.execute('INSERT INTO `megaproyecto`.`variable` (`id`, `id_variable`, `id_campo`, `valor`, `variablecol`) VALUES (%s, %s , 1, %s,1);', (var_count, person_count-1,sex))
          con.commit()
          var_count += 1

      demographics = getDemographic(edad, pebl,  dsmt, hare, ciep, cief, ciec, ciem, ciex, cies, cie)
      if demographics != 0:
          dem_count = 1
          for demo in demographics:
              print(demo)
              cur.execute('INSERT INTO `megaproyecto`.`variable` (`id`, `id_variable`, `id_campo`, `valor`, `variablecol`) VALUES (%s, %s , %s, %s,1);', (var_count, person_count-1, 1+dem_count ,demo))
              con.commit()
              var_count += 1
              dem_count += 1

      verdad = True
      contador = 1
      local_q_count = 1
      if verdad:
          cur.execute('INSERT INTO `megaproyecto`.`pregunta` (`id`, `id_persona`,`numero`,`veracidad`) VALUES (%s, %s,%s,%s);', (q_count, int(person_count), local_q_count, str(verdad)))
          con.commit()
      else:
          cur.execute('INSERT INTO `megaproyecto`.`pregunta` (`id`, `id_persona`,`numero`,`veracidad`) VALUES (%s, %s,%s,%s);', (q_count, int(person_count), local_q_count, str(verdad)))
          con.commit()
      local_q_count += 1
      second = int(second)
      print('Second:', second)
      print('LEN',len(af3))
      if second <= 140 and len(af3)>second:
          print('INSIDE IF ')
          cur.execute('INSERT INTO `megaproyecto`.`medicion` (`id`, `id_sensor`, `id_pregunta`, `segundo`, `medicion`) VALUES (%s, %s, %s, %s, %s);', (meassure_count, 1, q_count, second, af3[second]))
          con.commit()
          meassure_count += 1
          cur.execute('INSERT INTO `megaproyecto`.`medicion` (`id`, `id_sensor`, `id_pregunta`, `segundo`, `medicion`) VALUES (%s, %s, %s, %s, %s);', (meassure_count, 2, q_count, second, f3[second]))
          con.commit()
          meassure_count += 1
          cur.execute('INSERT INTO `megaproyecto`.`medicion` (`id`, `id_sensor`, `id_pregunta`, `segundo`, `medicion`) VALUES (%s, %s, %s, %s, %s);', (meassure_count, 3, q_count, second, af4[second]))
          con.commit()
          meassure_count += 1
          cur.execute('INSERT INTO `megaproyecto`.`medicion` (`id`, `id_sensor`, `id_pregunta`, `segundo`, `medicion`) VALUES (%s, %s, %s, %s, %s);', (meassure_count, 4, q_count, second, f4[second]))
          con.commit()
          meassure_count += 1
      print('OUT OF IF ')
      q_count += 1


def getTableQuestionVera(veracidad, preguntaNum):
  con = mysql.connector.connect(
    host="localhost",
    user="root",
    passwd="admin",
    database="megaproyecto"
  )

  cursor = con.cursor(buffered=True)
  af3 = []
  f3 = []
  af4 = []
  f4 = []
  sensores = [af3, f3,af4,f4]
  for sensorId in range(1,5):
      cursor.execute("SELECT medicion.medicion FROM medicion INNER JOIN pregunta ON pregunta.id = medicion.id_pregunta INNER JOIN sensor ON sensor.id = medicion.id_sensor INNER JOIN persona ON persona.idpersona = pregunta.id_persona WHERE pregunta.numero = "+str(preguntaNum)+" AND pregunta.veracidad = '"+veracidad+"' AND sensor.id = "+str(sensorId)+" ORDER BY persona.idpersona;")
      numrows = cursor.rowcount
      print (numrows)
      for x in range(0,numrows):
        row = cursor.fetchone()
        sensores[sensorId-1].append(row[0])

  sexo = []
  persona = []

  cursor.execute("SELECT variable.valor, persona.codigo FROM variable INNER JOIN variables ON variable.id_variable = variables.id INNER JOIN campo ON variable.id_campo = campo.id INNER JOIN persona ON persona.idpersona = variables.id_persona WHERE campo.nombre = 'Sexo' ORDER BY persona.idpersona;")
  numrows = cursor.rowcount
  for y in range(0,numrows):
      row = cursor.fetchone()
      sexo.append(row[0])
      persona.append(row[1])

  demographics = [[],[],[],[],[],[],[],[],[],[],[]]
  for campo_id in range(2,13):
    cursor.execute("SELECT variable.valor FROM variable INNER JOIN variables ON variable.id_variable = variables.id INNER JOIN campo ON variable.id_campo = campo.id INNER JOIN persona ON persona.idpersona = variables.id_persona WHERE campo.id = "+str(campo_id)+" ORDER BY persona.idpersona;")
    numrows = cursor.rowcount
    for num in range(0,numrows):
        row = cursor.fetchone()
        demographics[campo_id-2].append(str(row[0]))

  #demographics_values = str.format('{},{},{},{},{},{},{},{},{},{}',)
  output = ''

  print(sexo)
  print(demographics)
  print()
  for medicion in range(0,len(sensores[0])):
    try:
      str_dem = ""
      for p in range(0, 11):
        if p == 0:
          str_dem = str(demographics[p][medicion])
        else:
          str_dem = str(demographics[p][medicion])+ ','+ str_dem 
      if veracidad == 'True':
        output = output+ persona[medicion]+','+str(sensores[0][medicion])+','+str(sensores[1][medicion])+','+str(sensores[2][medicion])+','+str(sensores[3][medicion])+','+str(sexo[medicion])+","+str(preguntaNum)+',1,'+'2,'+str_dem+'\n'
      else:
        output = output+ persona[medicion]+','+str(sensores[0][medicion])+','+str(sensores[1][medicion])+','+str(sensores[2][medicion])+','+str(sensores[3][medicion])+','+str(sexo[medicion])+","+str(preguntaNum)+',0,'+'2,'+str_dem+'\n'
    except:
      print ('Oops')
  print (output)
  return output

def generate_file():
  full_output = 'persona,AF3,F3,AF4,F4,sexo,numPregunta,veracidad,escolaridad,cie,cies,ciex,ciem,ciec,cief,ciep,hare,dsmt,pebl,edad\n'
  full_output = full_output + str(getTableQuestionVera('True',1))

  with open('./result.csv', 'w') as file:
    print('writing')
    file.write(full_output)

def get_knn():
  df = pd.read_csv('./result.csv')
  df.drop(['persona','numPregunta','cie'], 1, inplace=True)

  x = np.array(df.drop(['veracidad'], 1))
  y = np.array(df['veracidad'])

  clf = neighbors.KNeighborsClassifier(n_neighbors=19)

  x_train2 = np.loadtxt('./train/x_train.txt')
  y_train2 = np.loadtxt('./train/y_train.txt')
  one_value = x
  print(one_value)
  print(x_train2[0])
  clf.fit(x_train2, y_train2)
  prediction = clf.predict(one_value)
  print('Accuracy of KNN: ',prediction)
  prediction = prediction[0]
  probability = clf.predict_proba(one_value)[0]
  return prediction, probability


class AudioRecorder(multiprocessing.Process):
  def __init__(self, ):
    multiprocessing.Process.__init__(self)
    self.exit = multiprocessing.Event()

  def run(self):
    print("On recordAudio")
    global recording
    CHUNK = 1024
    FORMAT = pyaudio.paInt16
    CHANNELS = 1
    RATE = 44100
    WAVE_OUTPUT_FILENAME = "output.wav"

    p = pyaudio.PyAudio()

    stream = p.open(format=FORMAT,
                    channels=CHANNELS,
                    rate=RATE,
                    input=True,
                    frames_per_buffer=CHUNK)

    frames = []
    print("* recording")

    while not self.exit.is_set():
      print("On recording loop")
      data = stream.read(CHUNK)
      frames.append(data)
    print("You exited!")

    print("* done recording")

    stream.stop_stream()
    stream.close()
    p.terminate()

    wf = wave.open(WAVE_OUTPUT_FILENAME, 'wb')
    wf.setnchannels(CHANNELS)
    wf.setsampwidth(p.get_sample_size(FORMAT))
    wf.setframerate(RATE)
    wf.writeframes(b''.join(frames))
    wf.close()

  def shutdown(self):
    print("Shutdown initiated")
    self.exit.set()


class EmotivRecorder(multiprocessing.Process):
  def __init__(self, ):
    multiprocessing.Process.__init__(self)
    self.exit = multiprocessing.Event()

  def run(self):
    print("On emotiv")
    os.system('c:\\Python27\python.exe .\Emotiv\Emotrix\emotrix\my_recorder.py ')
    self.exit.set()

class VideRecorder(multiprocessing.Process):
    def __init__(self, ):
        multiprocessing.Process.__init__(self)
        self.exit = multiprocessing.Event()

    def run(self):
        print("On Video Recorder...")
        lastIndex = 0
        files = glob.glob('./*.avi')
        if (len(files) > 0):
          lastIndex = int(max(files, key=os.path.getctime).split("-")[1].split(".")[0])
        lastIndex += 1
        cap = cv2.VideoCapture(0)
        frame_width = int(cap.get(3))
        frame_height = int(cap.get(4))
        out = cv2.VideoWriter('interview-%s.avi' % str(lastIndex), cv2.VideoWriter_fourcc(
            'M', 'J', 'P', 'G'), 30, (frame_width, frame_height))
        while(not self.exit.is_set()):
            ret, frame = cap.read()

            if ret == True:
                out.write(frame)

                cv2.imshow('frame', frame)

                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
            else:
                break
        cap.release()
        out.release()
        cv2.destroyAllWindows()

    def shutdown(self):
        self.exit.set()


app = Flask(__name__)
app.config['SECRET_KEY'] = 'secret!'
CORS(app)
socketio = SocketIO(app)
environment.reproducible()
global audioRecorder
global videoRecorder
global my_emotiv
global t0

def koch(second,sexo, edad, cie, dsmt, hare, ciep, cief, ciec, ciem, ciex=None, cies=None,pebl=None):
  if pebl is None:
    pebl = 0
  if ciex is None:
    ciex = 0
  if cies is None:
    cies = 0
  print('Running Koch')
  insert_BD(second, sexo, edad, pebl, dsmt, hare, ciep, cief, ciec, ciem, ciex, cies, cie)
  generate_file()
  value, confidence = get_knn()
  if value:
    category = 'true'
    confidence = confidence[1] * 100
  else:
    category = 'false'
    confidence = confidence[0] * 100
  time.sleep(3)
  requests.post('http://localhost:5000/send-koch-response', data={ "category": category, "confidence": confidence })
  
def leonel(gender, age, dsmt, hare, ciep, cief, ciec, ciem,cie):
  print('Running Leonel')
  age = float(age)
  dsmt = float(dsmt)
  hare = float(hare)
  ciep = float(ciep)
  cief = float(cief)
  ciec = float(ciec)
  ciem = float(cie)
  x = prediction(gender, age, dsmt, hare, ciep, cief, ciec, ciem,cie)
  requests.post('http://localhost:5000/send-leonel-response', data={ "category": x[0], "confidence": x[1] })

def chan(audioPath, cie, pebl, dsmt, hare):
  print('Running Chan')
  os.system('del chanSoX.txt')
  os.system('c:\\"Program Files (x86)"\sox-14-4-2\sox.exe ' + audioPath + ' −n stats >> chanSoX.txt 2>&1')
  with open('chanSoX.txt') as soxOutput:
    soxStats = soxOutput.readlines()
  rmsLevel = float(soxStats[4].rstrip()[10:])
  crest = float(soxStats[7].rstrip()[12:])
  rmsTr = float(soxStats[6].rstrip()[9:])

  db = MongoClient().voz.answers
  answers = list(db.find())
  rmsLevels = list(map(lambda answer: answer['rmsLevel'], answers))
  crests = list(map(lambda answer: answer['crest'], answers))
  rmsTrs = list(map(lambda answer: answer['rmsTr'], answers))
  cies = list(map(lambda answer: answer['cie'], answers))
  pebls = list(map(lambda answer: answer['pebl'], answers))
  dsmts = list(map(lambda answer: answer['dsmt'], answers))
  hares = list(map(lambda answer: answer['hare'], answers))

  maxrmsLevels = max(list(map(abs, rmsLevels)))
  maxcrests = max(list(map(abs, crests)))
  maxrmsTrs = max(list(map(abs, rmsTrs)))
  maxcies = max(list(map(abs, cies)))
  maxpebls = max(list(map(abs, pebls)))
  maxdsmts = max(list(map(abs, dsmts)))
  maxhares = max(list(map(abs, hares)))

  train_labels = np.array(list(map(lambda answer: answer['label'], answers)))
  rmsLevels = list(map(lambda number: number / maxrmsLevels, rmsLevels))
  crests = list(map(lambda number: number / maxcrests, crests))
  rmsTrs = list(map(lambda number: number / maxrmsTrs, rmsTrs))
  cies = list(map(lambda number: number / maxcies, cies))
  pebls = list(map(lambda number: number / maxpebls, pebls))
  dsmts = list(map(lambda number: number / maxdsmts, dsmts))
  hares = list(map(lambda number: number / maxhares, hares))
  train_answers = np.array([rmsLevels, crests, rmsTrs, cies, pebls, dsmts, hares])

  pnn = algorithms.PNN(std=0.25)
  pnn.train(train_answers.T, train_labels)
  rmsLevel = rmsLevel / maxrmsLevels
  crest = crest / maxcrests
  rmsTr = rmsTr / maxrmsTrs
  cie = float(cie) / maxcies
  pebl = float(pebl) / maxpebls
  dsmt = float(dsmt) / maxdsmts
  hare = float(hare) / maxhares
  test = np.array([[rmsLevel], [crest], [rmsTr], [cie], [pebl], [dsmt], [hare]])
  falseProbability, trueProbability = pnn.predict_proba(test.T)[0]
  if trueProbability > falseProbability:
    category = "true"
    confidence = trueProbability
  else:
    category = "false"
    confidence = falseProbability
  requests.post('http://localhost:5000/send-chan-response', data={ "category": category, "confidence": str(confidence*100) })

def olga():
  print('Running Olga')
  classif, fin_conf = return_values('output.wav')  
  requests.post('http://localhost:5000/send-olga-response', data={ "category": classif, "confidence": fin_conf })

def alvaro():
  print("Running Al")
  import data_manager_alvaro as dma
  import tensor_nonlinear_gaussian_rbf_SVM as tnlg
  dma.generateEntryFile("files_Al/test.csv") # aqui el path al archivo del emotiv
  pred, cert = tnlg.module("files_Al/input.csv")
  requests.post('http://localhost:5000/send-alvaro-response', data={ "category": pred, "confidence": cert })

def castro():
      print("Running Castro")
      classif, conf = expression()
      requests.post('http://localhost:5000/send-castro-response', data={ "category": classif, "confidence": conf })

@app.route('/start-question', methods=['POST'])
def startQuestion():
  print('Starting question')
  global videoRecorder
  global my_emotiv
  global t0
  t0 = int(time.time())
  my_emotiv = EmotivRecorder()
  my_emotiv.start()
  videoRecorder = VideRecorder()
  videoRecorder.start()
  return 'OK'

@app.route('/start-answer', methods=['POST'])
def startAnswer():
  print('Starting answer')
  global audioRecorder
  global t0
  t0 = int(time.time()-t0)
  audioRecorder = AudioRecorder()
  audioRecorder.start()
  return 'OK'

@app.route('/finish-answer', methods=['POST'])
def finishAnswer():
  global audioRecorder
  global my_emotiv
  audioRecorder.shutdown()
  global videoRecorder
  global t0
  videoRecorder.shutdown()
  requestJson = request.get_json()
  gender = requestJson['gender']
  age = requestJson['age']
  pebl = requestJson['pebl']
  dsmt = requestJson['dsmt']
  hare = requestJson['hare']
  ciep = requestJson['ciep']
  cief = requestJson['cief']
  ciec = requestJson['ciec']
  ciem = requestJson['ciem']
  cie = requestJson['cie']
  pool.apply_async(chan, ("output.wav",cie,pebl,dsmt,hare))
  pool.apply_async(olga)
  pool.apply_async(alvaro)
  pool.apply_async(leonel, (gender, age, dsmt, hare, ciep, cief, ciec, ciem,cie))
  pool.apply_async(castro)
  while my_emotiv.is_alive() :
    print('Alive')
    pass
  print('Exit emotiv')
  pool.apply_async(koch, (t0, gender, age, cie, dsmt, hare, ciep, cief, ciec, ciem))

  socketio.emit('started_analyzing')
  print('Finishing answer')
  return 'OK'

@app.route('/save-results', methods=['POST'])
def saveResults():
  requestJson = request.get_json()
  id = str(requestJson['id'])
  gender = str(requestJson['gender'])
  age = str(requestJson['age'])
  pebl = str(requestJson['pebl'])
  dsmt = str(requestJson['dsmt'])
  hare = str(requestJson['hare'])
  ciep = str(requestJson['ciep'])
  cief = str(requestJson['cief'])
  ciec = str(requestJson['ciec'])
  ciem = str(requestJson['ciem'])
  cie = str(requestJson['cie'])
  voiceModuleOlgaCategory = str(requestJson['voiceModuleOlgaCategory'])
  voiceModuleOlgaConfidence = str(requestJson['voiceModuleOlgaConfidence'])
  voiceModuleOlgaStatus = str(requestJson['voiceModuleOlgaStatus'])
  voiceModuleLeonelCategory = str(requestJson['voiceModuleLeonelCategory'])
  voiceModuleLeonelConfidence = str(requestJson['voiceModuleLeonelConfidence'])
  voiceModuleLeonelStatus = str(requestJson['voiceModuleLeonelStatus'])
  voiceModuleChanCategory = str(requestJson['voiceModuleChanCategory'])
  voiceModuleChanConfidence = str(requestJson['voiceModuleChanConfidence'])
  voiceModuleChanStatus = str(requestJson['voiceModuleChanStatus'])
  eegModuleKochCategory = str(requestJson['eegModuleKochCategory'])
  eegModuleKochConfidence = str(requestJson['eegModuleKochConfidence'])
  eegModuleKochStatus = str(requestJson['eegModuleKochStatus'])
  eegModuleRudyCategory = str(requestJson['eegModuleRudyCategory'])
  eegModuleRudyConfidence = str(requestJson['eegModuleRudyConfidence'])
  eegModuleRudyStatus = str(requestJson['eegModuleRudyStatus'])
  eegModuleAlvaroCategory = str(requestJson['eegModuleAlvaroCategory'])
  eegModuleAlvaroConfidence = str(requestJson['eegModuleAlvaroConfidence'])
  eegModuleAlvaroStatus = str(requestJson['eegModuleAlvaroStatus'])
  microModuleCastroCategory = str(requestJson['microModuleCastroCategory'])
  microModuleCastroConfidence = str(requestJson['microModuleCastroConfidence'])
  microModuleCastroStatus = str(requestJson['microModuleCastroStatus'])
  microModuleNoriegaCategory = str(requestJson['microModuleNoriegaCategory'])
  microModuleNoriegaConfidence = str(requestJson['microModuleNoriegaConfidence'])
  microModuleNoriegaStatus = str(requestJson['microModuleNoriegaStatus'])
  print('Results parsed')
  with open('results.csv', 'a') as results:
    result = id + ', ' + gender + ', ' + age + ', ' + pebl + ', ' + dsmt + ', ' + hare + ', ' + ciep + ', ' + cief + ', ' + ciec + ', ' + ciem + ', ' + cie + ', ' + voiceModuleOlgaCategory + ', ' + voiceModuleOlgaConfidence + ', ' + voiceModuleOlgaStatus + ', ' + voiceModuleLeonelCategory + ', ' + voiceModuleLeonelConfidence + ', ' + voiceModuleLeonelStatus + ', ' + voiceModuleChanCategory + ', ' + voiceModuleChanConfidence + ', ' + voiceModuleChanStatus + ', ' + eegModuleKochCategory + ', ' + eegModuleKochConfidence + ', ' + eegModuleKochStatus + ', ' + eegModuleRudyCategory + ', ' + eegModuleRudyConfidence + ', ' + eegModuleRudyStatus + ', ' + eegModuleAlvaroCategory + ', ' + eegModuleAlvaroConfidence + ', ' + eegModuleAlvaroStatus + ', ' + microModuleCastroCategory + ', ' + microModuleCastroConfidence + ', ' + microModuleCastroStatus + ', ' + microModuleNoriegaCategory + ', ' + microModuleNoriegaConfidence + ', ' + microModuleNoriegaStatus + '\n'
    results.write(result)
  return 'OK'

@app.route('/send-olga-response', methods=['POST'])
def sendOlgaResponse():
  socketio.emit('olga_response', { 'data': request.form })
  print('Sent olga message')
  return 'OK'

@app.route('/send-leonel-response', methods=['POST'])
def sendLeonelResponse():
  socketio.emit('leonel_response', { 'data': request.form })
  print('Sent leonel message')
  return 'OK'

@app.route('/send-chan-response', methods=['POST'])
def sendChanResponse():
  socketio.emit('chan_response', { 'data': request.form })
  print('Sent chan message')
  return 'OK'

@app.route('/send-koch-response', methods=['POST'])
def sendKochResponse():
  socketio.emit('koch_response', { 'data': request.form })
  print('Sent koch message')
  return 'OK'

@app.route('/send-rudy-response', methods=['POST'])
def sendRudyResponse():
  socketio.emit('rudy_response', { 'data': request.form })
  print('Sent rudy message')
  return 'OK'

@app.route('/send-alvaro-response', methods=['POST'])
def sendAlvaroResponse():
  socketio.emit('alvaro_response', { 'data': request.form })
  print('Sent alvaro message')
  return 'OK'

@app.route('/send-castro-response', methods=['POST'])
def sendCastroResponse():
  socketio.emit('castro_response', { 'data': request.form })
  print('Sent castro message')
  return 'OK'

@app.route('/send-noriega-response', methods=['POST'])
def sendNoriegaResponse():
  socketio.emit('noriega_response', { 'data': request.form })
  print('Sent noriega message')
  return 'OK'

if __name__ == '__main__':
  pool = multiprocessing.Pool(processes=2)
  socketio.run(app)
