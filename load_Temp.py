import re
import numpy as np
import time
import os


# Rize La:41  Long:40
# Brest La 48  Long:356 ( 356 = -4 % 360)
# Paris la 48 Long: 2

# Range of latitude(40 -- 50)(10 slots)
# Range of Longitude(355 --359,,0 -- 40)(46 slots)

# RE expression matching patten
patten =re.compile(r"\S\d*\.\d*")

# UTC time
years = ['2017','2018']
mounths = ['0'+str(i) for i in range(1,10)]+[str(i) for i in range(10,13)]
days = ['0'+str(i) for i in range(1,10)] + [str(i) for i in range(10,32)]
hours = ['00','06','12','18']

print('length of days',len(days))
print('length of mounth',len(mounths))

path = "D:\IMT_Atlantique\S4\ELU501\Challenge3\\temperatures"
all = []
pre = time.time()
numFile = 0
for year in years:
    print('year:',year)
    for mounth in mounths:
        print('mounth ',mounth)
        for day in days:
            print('data',day,' ',mounth,' ',year,end='--')
            # print(' Only for one day')
            a_day = []
            for hour in hours:
                a = []
                a_hour = []
                file_name = path + '\gfsanl_3_' + year + mounth + day + '_' + hour + '00.txt'
                if not os.path.exists(file_name):
                    continue
                f = open(file_name,'r')
                lines = f.readlines()
                for line in lines:
                    a = []
                    list = line.strip('\n').split(' ')
                    for i in list:
                        match = patten.match(i)
                        if match:
                            a.append(float(match.group()))
                    if a and len(a) == 3 and 40 <= a[0] < 50 and (a[1] <= 40 or a[1] >= 355):
                        if a[2]*100-273.15 > 18:
                            t = 18
                        elif a[2]*100-273.15 < 0:
                            t = 0
                        else:
                            t = a[2]*100-273.15
                        a_hour.append((a[0], a[1], t))
                numFile += 1
                if numFile % 10 == 0:
                    print(numFile, end='--')
                    print('read time is ', time.time() - pre,end = "//")
                f.close()
                if a_hour:
                    a_day.append(a_hour)
            print("length in a day: ",len(a_day))
            if a_day:
                np_a_day = np.array(a_day)
                x,y,z = np_a_day.shape
                average_t = np.zeros((y,z))
                for i in range(x):
                    # print(np_a_day)
                    # print(file_name)
                    np_a_day[0, :, 2] = np_a_day[0,:,2] + np_a_day[i,:,2]
                average_t[:,:2] = np_a_day[0,:,:2]
                average_t[:,2] = np_a_day[0,:,2]/3
                all.append(average_t)


print('length of data',numFile)
np_a = np.array(all)
print(np_a.shape)
print('Read ',numFile,' files')
print('read file takes',time.time()-pre,' s')

# save the useful data
np.save("Temperature_Rize2Brest.npy",np_a)
# read data as this form b = np.load("Temperature_Rize2Brest.npy")