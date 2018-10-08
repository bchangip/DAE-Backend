from flask import Flask, request
from flask_socketio import SocketIO, emit, send
from flask_cors import CORS
import time
import multiprocessing
import requests


app = Flask(__name__)
app.config['SECRET_KEY'] = 'secret!'
CORS(app)
socketio = SocketIO(app)

def koch():
  print('Running Koch')
  time.sleep(3)
  requests.post('http://localhost:5000/send-koch-response', data={ "category": "false", "confidence": 82 })

def chan():
  print('Running Chan')
  time.sleep(5)
  requests.post('http://localhost:5000/send-chan-response', data={ "category": "true", "confidence": 79 })

@app.route('/start-question')
def startQuestion():
  print('Starting question')

@app.route('/start-answer')
def startAnswer():
  print('Starting answer')

@app.route('/finish-answer')
def finishAnswer():
  pool.apply_async(chan)
  pool.apply_async(koch)
  socketio.emit('started_analyzing')
  print('Finishing answer')
  return 'OK'

@app.route('/send-chan-response', methods=['POST'])
def sendChanResponse():
  print('Chan response', request.form)
  socketio.emit('chan_response', { 'data': request.form })
  print('Sent chan message')
  return 'OK'

@app.route('/send-koch-response', methods=['POST'])
def sendKochResponse():
  print('Koch response', request.form)
  socketio.emit('koch_response', { 'data': request.form })
  print('Sent koch message')
  return 'OK'

if __name__ == '__main__':
  pool = multiprocessing.Pool(processes=20)
  socketio.run(app)
