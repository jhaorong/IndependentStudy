import torch
import torch.nn as nn
import torch.utils.data as Data
import numpy as np
import pandas as pd
import time
from sklearn import preprocessing
batch_size = 100
MAXCACHESIZE = 4000
f_input = open("input.txt", "r")
first_line = f_input.readline()  # store first line label
input_lines = f_input.readlines()  # Store all data each row in a list
numberOfLines = len(input_lines)
requestsIdList = [] #A list to store request id(int,begin from 0)
requestTimeList = []  # A list to store requestTime(float type)
sectorNumberList = []  # A list to store SectorNumber(int type)
past_countlist = []         #A list to store currently request's request_count in the past
stayCacheTimeList = [0]*numberOfLines #A list to store how long is the request stay in cache
hitList = [0]*numberOfLines
# get train data
#i=0
for id,line in enumerate(input_lines):
    formLine = line.split('\t')  # split the line with tab
    #print(formLine)
    requestsIdList.append(id)
    requestTimeList.append(float(formLine[0]))
    sectorNumberList.append(int(formLine[1]))
    past_countlist.append(int(formLine[2]))
    #print(requestTimeList[i],"\t",sectorNumberList[i],"\t",past_countlist[i],"\n")
    #i+=1

#Calculate how long does request stay in SSD
currentCacheId = 0
index = 0
hitcount = 0
SSD_DT = {} #{sectornumber:request_id}
hit_DT = {} #{request_id:sectornumber}
i=0
j=0
for requestId in range(numberOfLines):
    print("request id = ",requestId)
    if str(sectorNumberList[requestId]) in SSD_DT.keys():               # hit
        print("SSD is hit\n")
        hitList[requestId] = 1
        SSD_reid = SSD_DT[str(sectorNumberList[requestId])]
        #print("ssd staycachetime = ",stayCacheTimeList[SSD_reid])
        stayCacheTimeList[requestId] = stayCacheTimeList[SSD_reid]      #hit staycachetime = ssd staycachetime
        hit_DT[str(requestId)] = sectorNumberList[requestId]            #record hit request_id,sectornumber
        hitcount += 1
    else:  # not hit
        if currentCacheId < MAXCACHESIZE:  # SSD is available
            print("SSD is available\n")
            #SSD[currentCacheId] = sectorNumberList[requestId]
            SSD_DT[str(sectorNumberList[requestId])] = requestId        #sectornumber:request_id
            currentCacheId += 1
        else:                              #SSD is full
            #print(type(SSD_DF['past_count']))
            min = 9223372036854775807
            minsec = 0
            print("SSD is full\n")
            for reid in SSD_DT.values():
                if past_countlist[reid] < min:
                    min = past_countlist[reid]
                    min_sec = sectorNumberList[reid]    #find minimum benefit value's index in SSD_DF
            #print("min_index = ",min_index)
            del SSD_DT[str(min_sec)]
            SSD_DT[str(sectorNumberList[requestId])] = requestId
    for reid in SSD_DT.values():
        stayCacheTimeList[reid] += 1
    for li in list(hit_DT.items()):#request_id : sectornumber
        if str(li[1]) in SSD_DT.keys():
            stayCacheTimeList[int(li[0])] += 1
        else:
           del hit_DT[str(li[0])]

    #print("間格b = ",end_time-start_time)

f_input_all = open("input_all.txt","w")
f_input_all.write("request_time\tsector_number\tpast_count\tstay_Cache_Time\thit\n")
for i in range(numberOfLines):
    f_input_all.write("{}\t{}\t{}\t{}\n".format(requestTimeList[i],sectorNumberList[i],past_countlist[i],stayCacheTimeList[i],hitList[i]))

