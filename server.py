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

app = Flask(__name__)
app.config['SECRET_KEY'] = 'secret!'
CORS(app)
socketio = SocketIO(app)
environment.reproducible()
global audioRecorder

def koch():
  print('Running Koch')
  time.sleep(3)
  requests.post('http://localhost:5000/send-koch-response', data={ "category": "false", "confidence": 82 })

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

@app.route('/start-question', methods=['POST'])
def startQuestion():
  print('Starting question')
  return 'OK'

@app.route('/start-answer', methods=['POST'])
def startAnswer():
  print('Starting answer')
  global audioRecorder
  audioRecorder = AudioRecorder()
  audioRecorder.start()
  return 'OK'

@app.route('/finish-answer', methods=['POST'])
def finishAnswer():
  global audioRecorder
  audioRecorder.shutdown()
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
  pool.apply_async(koch)
  socketio.emit('started_analyzing')
  print('Finishing answer')
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
  pool = multiprocessing.Pool(processes=10)
  socketio.run(app)
