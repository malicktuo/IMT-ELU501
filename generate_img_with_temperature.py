import pickle
import numpy as np
import PIL as im
from matplotlib import pyplot as plt

# Generate image with shortest path 

input_filename = "D:\IMT_Atlantique\S4\ELU501\Challenge3\population-density-map.bmp"
img = im.Image.open(input_filename)
G_orig = np.array(img)

dic = open("previous_zombie_with_t.pkl",'rb')
previous = pickle.load(dic)
dic.close()
x1 = 1280
y1 = 600
## Cities' coordinates
paris = (1250-x1,1200-y1)
renne = (1337-x1,888-y1)
brest = (1306-x1,669-y1)
destinations = [brest,renne,paris]

start = (2108, 4426)
np_img = np.int32(np.array(img))
no_previous = []

for des in destinations:
    path = []
    now = des
    while now != start:
        if now in previous.keys():
            pre = previous[now]
            path.append(pre)
            now = pre
        else:
            no_previous.append(now)
            break
    print('length of the path is', len(path))
    #### Paris (1250,1200)
    #### Londre (1000,1000)
    #### Renne (1337,888)
    for i in range(len(path)):
        np_img[path[i][0]+x1, path[i][1]+y1] = (255, 0, 0)

plt.figure(dpi=3000)
plt.imshow(np_img)
plt.axis('off')
plt.savefig('test_temp.png')
plt.show()