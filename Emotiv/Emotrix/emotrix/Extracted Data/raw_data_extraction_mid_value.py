import csv
import sys
with open(sys.argv[1]+'.csv', 'rb') as csvfile:
	dataInput = csv.reader(csvfile, delimiter=' ', quotechar='|')
	af3 = ''
	count_af3 = 0
        current_sec = 0
        current_count = 0
        current_sum = 0
        data = []
	for row in dataInput:
		line = ''.join(row)
		splitData = line.split('\t')
		#print 'Second of Data ' + str(splitData[0])
		#Using sensors: AF3, AF4, F3 and F4
		for i in range(1, len(splitData)-1):
			splitDots= splitData[i].split(":")
			splitCommas = splitDots[1].split(',')
			if splitDots[0] == sys.argv[2] :
                                if current_sec == int(splitData[0]) :
                                        data.append(int(splitCommas[1]))
                                        current_count += 1
                                else:
                                        print 'change sec'
                                        if (current_count % 2 == 0):
                                                half = current_count / 2
                                                value = (data[half] + data[half+1])/2
                                                af3 = af3 + str(count_af3) + ',' + str(value) + '\n'
                                                count_af3 = count_af3+1
                                        else:
                                                middle = data[int(current_count / 2 + 0.5)]
                                                af3 = af3 + str(count_af3) + ',' + str(middle) + '\n'
                                                count_af3 = count_af3+1
                                        current_sec = int(splitData[0])
                                        data = []
                                        data.append(int(splitCommas[1]))
                                        current_count= 0
                                        
        with open(sys.argv[2]+'.csv','wb') as file:
                file.write(af3)
