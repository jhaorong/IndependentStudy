MAXCACHESIZE = 4000
def read_tracedata(filename):
    f = open(filename,"r")      #open the trace file
    output = open("output2.txt", "w")    #open the output file
    output.write("Request time\tSectorNumber\tCache(1 success 0 fail)\tHit(1 success 0 fail)\tfuture_write_count\tpast_write_count\n") #output requestTime   SectorNumber    Cache
    file_lines = f.readlines()                                            #Store all data each row in a list
    numberOfLines = len(file_lines)                                       #Store file number of row
    index=0
    maxSectorNumber = 0
    SSD = [0] * MAXCACHESIZE  # A list to store content in SSD Cache
    requestTimeList = []  # A list to store requestTime
    sectorNumberList = []  # A list to store SectorNumber
    currentCachenum = 0
    SSDindex = 0
    hitcount = 0
    for line in file_lines:
        line = line.strip()                 #delete the head and tail with space or newline
        formLine = line.split('\t')         #split the line with tab
        requestTimeList.append(formLine[0])
        sectorNumberList.append(formLine[2])
        sectorNumberToInt = int(sectorNumberList[index])
        if sectorNumberToInt > maxSectorNumber: # Find max SectorNumber
           maxSectorNumber = sectorNumberToInt
        index += 1

    future_count = [0] * maxSectorNumber                       #record the all data SectorNumber count
    past_count = [0] * maxSectorNumber
    for i in range(len(sectorNumberList)):
        sectorNumberToInt = int(sectorNumberList[i])
        future_count[sectorNumberToInt-1] += 1
    """
    #write count to txt
    duplicatenum = 0
    total = 0
    f2 = open("future_count.txt", "w")
    f2.write("sectornumber\tcount\n")
    for i in range(len(count)):
        if count[i] > 1:
            total += count[i]
            duplicatenum += 1
            f2.write(str(i+1)+"\t"+str(count[i])+"\n")
    print(f"total = {total} duplicatenum = {duplicatenum} duplicate count = {total - duplicatenum}\n")
    """

    cache = 0
    hit = 0
    for i in range(len(sectorNumberList)):
        sectorNumberToInt = int(sectorNumberList[i])
        future_count[sectorNumberToInt - 1] -= 1
        for j in range(currentCachenum):                   #Search Cache for Sectornumber
            if sectorNumberToInt == SSD[j]:                 #Find Sectornumber,hit
                hit = 1
                hitcount+=1
                cache = 0
                output.write(f"{requestTimeList[i]}\t{sectorNumberToInt}\t{cache}\t{hit}\t{future_count[sectorNumberToInt-1]}\t{past_count[sectorNumberToInt-1]}\n")
                break
            else:
                hit = 0
        if hit == 0:                                        #Not Find Sectornumber,hit fault
            if currentCachenum < MAXCACHESIZE:              # Cache is available
                SSD[currentCachenum] = sectorNumberToInt
                cache = 1                                   #Put into SSD
                output.write(f"{requestTimeList[i]}\t{sectorNumberToInt}\t{cache}\t{hit}\t{future_count[sectorNumberToInt-1]}\t{past_count[sectorNumberToInt-1]}\n")
                #print(f"cache {i} is set to  {sectorNumberToInt}")
                currentCachenum += 1
            else:                                           # Cache is full
                minCount = future_count[SSD[0]-1]                  #initial to first cache count
                minSectorNumber = SSD[0]                    #initial to first cache number
                for j in range(MAXCACHESIZE):               # Find minimun countNumber
                    countIndex = SSD[j] - 1
                    if future_count[countIndex] < minCount:
                        minCount = future_count[SSD[j] - 1]
                        minSectorNumber = SSD[j]
                for j in range(MAXCACHESIZE):
                    if SSD[j] == minSectorNumber:
                        if future_count[sectorNumberToInt-1] > minCount:   #if the request count > SSD minimum countNumber,put into cache
                            #print(f"Cache {j} and Sectornumber {SSD[j]} is replaced with {sectorNumberToInt}")
                            SSD[j] = sectorNumberToInt
                            #out = 1
                            cache = 1
                            #hitcount += 1
                            output.write(f"{requestTimeList[i]}\t{sectorNumberToInt}\t{cache}\t{hit}\t{future_count[sectorNumberToInt-1]}\t{past_count[sectorNumberToInt-1]}\n")
                            break
                        else:
                            #out = 0
                            #print(f"Sectornumber {sectorNumberToInt} don't cache in SSD")
                            cache = 0
                            output.write(f"{requestTimeList[i]}\t{sectorNumberToInt}\t{cache}\t{hit}\t{future_count[sectorNumberToInt-1]}\t{past_count[sectorNumberToInt-1]}\n")
                            break
        past_count[sectorNumberToInt - 1] += 1
    print(f"hit count = {hitcount} numberofline = {numberOfLines}")
    #output.write(f"hit ratio  {hitcount/numberOfLines}")
read_tracedata("cut_iozone_234204.txt")
