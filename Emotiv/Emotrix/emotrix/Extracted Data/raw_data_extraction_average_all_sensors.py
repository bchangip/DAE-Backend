import csv
import sys
dataAF3 = []
dataF3 = []
dataF4 = []
dataAF4 = []
output = ''
with open(sys.argv[1]+'.csv', 'rb') as csvfile:
	dataInput = csv.reader(csvfile, delimiter=' ', quotechar='|')
	af3 = ''
	count_af3 = 0
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
			if splitDots[0] == 'AF3' :
                                if current_sec == int(splitData[0]) :
                                        current_sum = current_sum + int(splitCommas[1])
                                        current_count += 1
                                else:
                                        average = current_sum / current_count
                                        current_sec = int(splitData[0])
                                        current_sum = int(splitCommas[1])
                                        current_count= 0
                                        dataAF3.append(average)
                                        count_af3 = count_af3+1
with open(sys.argv[1]+'.csv', 'rb') as csvfile:
	dataInput = csv.reader(csvfile, delimiter=' ', quotechar='|')
	af3 = ''
	count_af3 = 0
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
			if splitDots[0] == 'AF4' :
                                if current_sec == int(splitData[0]) :
                                        current_sum = current_sum + int(splitCommas[1])
                                        current_count += 1
                                else:
                                        average = current_sum / current_count
                                        current_sec = int(splitData[0])
                                        current_sum = int(splitCommas[1])
                                        current_count= 0
                                        dataAF4.append(average)
                                        count_f3 = count_af3+1 
with open(sys.argv[1]+'.csv', 'rb') as csvfile:
	dataInput = csv.reader(csvfile, delimiter=' ', quotechar='|')
	af3 = ''
	count_af3 = 0
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
			if splitDots[0] == 'F4' :
                                if current_sec == int(splitData[0]) :
                                        current_sum = current_sum + int(splitCommas[1])
                                        current_count += 1
                                else:
                                        average = current_sum / current_count
                                        current_sec = int(splitData[0])
                                        current_sum = int(splitCommas[1])
                                        current_count= 0
                                        dataF4.append(average)
                                        count_af3 = count_af3+1 
with open(sys.argv[1]+'.csv', 'rb') as csvfile:
	dataInput = csv.reader(csvfile, delimiter=' ', quotechar='|')
	af3 = ''
	count_af3 = 0
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
			if splitDots[0] == 'F3' :
                                if current_sec == int(splitData[0]) :
                                        current_sum = current_sum + int(splitCommas[1])
                                        current_count += 1
                                else:
                                        average = current_sum / current_count
                                        current_sec = int(splitData[0])
                                        current_sum = int(splitCommas[1])
                                        current_count= 0
                                        dataF3.append(average)
                                        count_af3 = count_af3+1 
print (dataAF3)
print (dataAF4)
print (dataF3)
print (dataF4)
for i in range(0, len(dataAF3)-1):
        output = output + str(dataAF3[i]) +','+str(dataAF4[i])+','+str(dataF3[i])+','+str(dataF4[i])+','
output = output + '0'
with open(sys.argv[1]+'_all_sensors.csv','wb') as file:
        file.write(output)
