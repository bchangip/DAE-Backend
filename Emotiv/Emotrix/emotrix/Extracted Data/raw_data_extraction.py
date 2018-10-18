import csv 
with open('A001 mentira.csv', 'rb') as csvfile:
	dataInput = csv.reader(csvfile, delimiter=' ', quotechar='|')
	af3 = ''
	count_af3 = 0
	af4 = ''
	count_af4 = 0
	f3 = ''
	count_f3 = 0
	f4 = ''
        count_f4 = 0
	for row in dataInput:
		line = ''.join(row)
		splitData = line.split('\t')
		#print 'Second of Data ' + str(splitData[0])
		#Using sensors: AF3, AF4, F3 and F4
		for i in range(1, len(splitData)-1):
			splitDots= splitData[i].split(":")
			splitCommas = splitDots[1].split(',')
			if splitDots[0] == "AF3" :
                                af3 = af3 + str(count_af3) + ',' + splitCommas[1] + '\n'
                                count_af3 = count_af3+1
                        if splitDots[0] == "AF4" :
                                af4 = af4 + str(count_af4) + ',' + splitCommas[1] + '\n'
                                count_af4 = count_af4+1
                        if splitDots[0] == "F3" :
                                f3 = f3 + str(count_f3) + ',' + splitCommas[1] + '\n'
                                count_f3 = count_f3+1
                        if splitDots[0] == "F4" :
                                f4 = f4 + str(count_f4) + ',' + splitCommas[1] + '\n'
                                count_f4 = count_f4+1
        with open('af3.csv','wb') as file:
                file.write(af3)
        with open('af4.csv','wb') as file:
                file.write(af4)
        with open('f3.csv','wb') as file:
                file.write(f3)
        with open('f4.csv','wb') as file:
                file.write(f4)
