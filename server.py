from flask import Flask, request
from flask_socketio import SocketIO, emit, send
from flask_cors import CORS
import time
import multiprocessing
import requests
import pyaudio
import wave

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
global audioRecorder

def koch():
  print('Running Koch')
  time.sleep(3)
  requests.post('http://localhost:5000/send-koch-response', data={ "category": "false", "confidence": 82 })

def chan(audioPath):
  print('Running Chan')
  print('Analyzing audio located at', audioPath)
  time.sleep(5)
  requests.post('http://localhost:5000/send-chan-response', data={ "category": "true", "confidence": 79 })

@app.route('/start-question')
def startQuestion():
  print('Starting question')
  return 'OK'

@app.route('/start-answer')
def startAnswer():
  print('Starting answer')
  global audioRecorder
  audioRecorder = AudioRecorder()
  audioRecorder.start()
  return 'OK'

@app.route('/finish-answer')
def finishAnswer():
  global audioRecorder
  audioRecorder.shutdown()
  pool.apply_async(chan, ("output.wav",))
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
  pool = multiprocessing.Pool(processes=20)
  socketio.run(app)
