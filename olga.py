import sox
import pickle


#get stats from audio file
def get_file_ready(path):
	tfm = sox.Transformer()
	return tfm.stats(path)

#get important variables 
def get_variables(dictionary):
	min_level = float(dictionary.get("Min level"))
	max_level = float(dictionary.get("Max level"))
	rms_level = float(dictionary.get("RMS lev dB"))
	pk_level = float(dictionary.get("Pk lev dB"))
	rms_tr = float(dictionary.get("RMS Tr dB"))
	crest = float(dictionary.get("Crest factor"))
	rms_pk = float(dictionary.get("RMS Pk dB"))
	return [[rms_tr,pk_level,crest,rms_pk,min_level,max_level,rms_level]]

def get_model(filename):
	with open(filename, 'rb') as file:
		pickle_model = pickle.load(file)
		print(pickle_model)
		return pickle_model

def return_values(path):
	a = get_file_ready(path)
	to_predict = get_variables(a)
	model = get_model('pickle_model_olga.pkl')
	pickle_model = model.predict(to_predict)
	rt =""
	if(pickle_model == 1.):
		rt="true"
	else:
		rt ="false"
	conf = model.predict_proba(to_predict)
	fin_conf = round((conf[0][0])*100,2)
	return rt, fin_conf
