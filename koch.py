import time
import multiprocessing
import os
import subprocess
global my_emotiv

class EmotivRecorder():

  def run(self):
    print("On emotiv")
    os.system('c:\\Python27\python.exe ..\Emotiv\Emotrix\emotrix\my_recorder.py ')
    print ('ID: ', os.getpid())

  def shutdown(self):
    print("Shutdown initiated")
    os.system('^C')

def startQuestion():
  print('Starting question')
  global my_emotiv
  my_emotiv = EmotivRecorder()
  my_emotiv.run()
  return 'OK'

def startAnswer():
  print('Starting answer')
  global my_emotiv
  my_emotiv.shutdown()
  return 'OK'

def finishAnswer():
  global audioRecorder
  audioRecorder.shutdown()
  pool.apply_async(chan, ("output.wav", 0.5, 0.12, 0.34, 0.89))
  pool.apply_async(koch)
  socketio.emit('started_analyzing')
  print('Finishing answer')
  return 'OK'

# subprocess.call(['py', '-2', '..\Emotiv\Emotrix\emotrix\my_recorder.py'])
# os.system('c:\\Python27\python.exe ..\Emotiv\Emotrix\emotrix\my_recorder.py ')
startQuestion()
time.sleep(6)
# startAnswer()
# subprocess.call(['^C'])