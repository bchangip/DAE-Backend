import csv
import sys
sensors = ['AF3','AF4','F3','F4']
for sensor in sensors:
        with open('emotiv.csv', 'rb') as csvfile:
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
        			if splitDots[0] == sensor :
                                        if current_sec == int(splitData[0]) :
                                                current_sum = current_sum + int(splitCommas[1])
                                                current_count += 1
                                        else:
                                                average = current_sum / current_count
                                                current_sec = int(splitData[0])
                                                current_sum = int(splitCommas[1])
                                                current_count= 0
                                                af3 = af3 + str(count_af3) + ',' + str(average) + '\n'
                                                count_af3 = count_af3+1
                with open(sensor+'.csv','wb') as file:
                        file.write(af3)         
                                    
