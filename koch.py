import time
import multiprocessing
import requests
import os
import xlrd
import csv
import sys
import mysql.connector
import pandas as pd
import numpy as np

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
      second = int(second) - 2
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

class EmotivRecorder():
  def __init__(self, ):
    multiprocessing.Process.__init__(self)
    self.exit = multiprocessing.Event()

  def run(self):
    print("On emotiv")
    os.system('c:\\Python27\python.exe .\Emotiv\Emotrix\emotrix\my_recorder.py ')
    self.exit.set()

global my_emotiv
global t0

def koch(second,sexo, edad, pebl, dsmt, hare, ciep, cief, ciec, ciem, ciex, cies, cie):
  print('Running Koch')
  insert_BD(second,sexo, edad, pebl,  dsmt, hare, ciep, cief, ciec, ciem, ciex, cies, cie)
  generate_file()
  value, confidence = get_knn()
  if value:
    category = 'true'
    confidence = confidence[1] * 100
  else:
    category = 'false'
    confidence = confidence[0] * 100
  time.sleep(3)
  print('category: ', category)
  print('confidence: ', confidence)
  requests.post('http://localhost:5000/send-koch-response', data={ "category": category, "confidence": confidence })

def startAnswer():
  print('Starting answer')
  global t0
  t0 = int(time.time()-t0)
  return 'OK'

def finishAnswer():
  global t0
  koch(t0,'Masculino', 23, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10)
  

def startQuestion():
  print('Starting question')
  global my_emotiv
  global t0
  t0 = int(time.time())
  my_emotiv = EmotivRecorder()
  my_emotiv.run()
  return 'OK'
