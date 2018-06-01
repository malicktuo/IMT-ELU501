from typing import List, Any, Union, Tuple

import PIL as im
import numpy as np
from matplotlib import pyplot as plt
import time
from heapq import heappop as pop
from heapq import heappush as push
from random import shuffle


def findIndexInMap(g_map, val):
    """
    # Function to search where the value is in an image
    #  input : g_map: image. type:ndarray
    #   		 val: value which we need to localize in the image.
    #				  type: int or float
    #  output: (x,y),a tulple of list which content the coordinates of the elements who equal val
    """
    mask = np.all(g_map.reshape(-1, 1) == val, axis=1)
    b = np.where(mask.reshape(g_map.shape))
    print(g_map[b[0], b[1]])
    return b[0], b[1]

def preprocessing():
    """
    This function is for preparing the data
    output: g_normalised : ndarray, normalised map, element will represent the the weight
            start and end : points in form of tuple (x,y)
            img: ndarrat, loaded image in black and white

    """

    # read image RGB from this directory
    input_filename = "D:\IMT_Atlantique\S4\ELU501\Challenge3\population-density-map.bmp"
    img = im.Image.open(input_filename)

    # generate a numpy array to represent image
    G_orig = np.array(img)

    # Reduce the scale of Image
    # x1 = 980
    # x2 = 2200
    # y1 = 600
    # y2 = 4450
    # G_orig = G_orig[x1:x2,y1:y2]

    # Find start point whose color is green and end point with red
    mask = np.all(G_orig == (255, 0, 0), axis=-1)
    endPoint = np.where(mask)
    mask = np.all(G_orig == (0, 255, 0), axis=-1)
    startPoint = np.where(mask)
    # start:G_orig[2108,4426]   G_orig[startPoint[0],startPoint[1]]
    # end: G_orig[1306,669]    G_orig[endPoint[0],endPoint[1]]

    # Convert the RGB image to black-white image
    img_gray = img.convert("L")
    G_gray = np.float32((np.array(img_gray)))

    # reduce the scale in order to avoid useless calculate
    # G_gray = G_gray[x1:x2,y1:y2]

    ###****** First normalization:
    #   Normalize the values of all elements of the ndarray between 0 and 1
    g_normalised = (G_gray - (np.min(G_gray))) / (np.max(G_gray) - np.min(G_gray))
    print("---First normalisation---")
    print("Values are the density of population")
    print(np.unique(g_normalised, return_counts=True))

    ### Second normalization:
    #   Change the values representing the population into the values representing
    #          the time which zombies need in order to cross this pixel.
    g_normalised = 1/((23/24) * g_normalised + (1/24))
    g_normalised = np.int32(np.round(g_normalised))
    print("---Second normalisation---")
    print("Values are the time which zombies need to cross the pixel")
    print(np.unique(g_normalised, return_counts=True))

    return g_normalised, startPoint[0][0], startPoint[1][0], endPoint[0][0], endPoint[1][0], G_orig


def drawImg(np_img, dpi,imgName):
    plt.figure(dpi=dpi)
    plt.imshow(np_img)
    plt.axis('off')
    plt.savefig(imgName)
    plt.show()


###################################################################################################
# ----- Here is the main function -----------

# SOME BIG CITIES :
# Paris (1250,1200)
# Londre (1000,1000)
# Renne (1337,888)

# create map :g_map , values in the map means hours to pass this pixel
# g_map is the map in which elements represent the time which zombie need to pass the pixel
# Define the start and end point
# np_img is used for showing the shortest path at the end of the code.
pre_pro = time.time()

# preprocessing() will prepare the map and start point and end point and also the loaded image
g_map, x_start, y_start, x_end, y_end, img, = preprocessing()

print('-- In New Graph --- ')
print('startPoint is ', (x_start, y_start))
print('endPoint is ', (x_end, y_end))

##################################################################################################
### Use Dijkstra algo to find the shortest path between the startPoint and other pixel
## Variable engaged
# Start and end point
start = (x_start, y_start)
end = (x_end, y_end)
shape = g_map.shape

# Minimum heap, which will pop the smallest value 
heap = []

# Find the previous pixel by the current pixel {current pixel: previous pixel}
previous = {}

# Distance between start node and others  {end:distance}
distances = {}

# the pixel which are not reachable
forbidden_zone = []

## Preperation for the algo
# fist position is start point and push start point into the heap 
now = start
push(heap, (g_map[start], start))

## Reset distance as 'inf' between the start point and others
for x in range(shape[0]):
    for y in range(shape[1]):
        distances[(x, y)] = float('inf')
distances[start] = 0
a = time.time()
print('\n Preprocessing takes ', a - pre_pro, 's \n')

b = a

# ### Localize where the zombies will die
# # make 2 windows in order to find zones with no population in 10km
# # Horizontal zone
# wds1 = np.zeros((1, 10))
#
# # Vertical Zone
# wds2 = np.zeros((10, 1))
# # wds2 = np.int32(wds2)
# # fulfill two matrix with zeros!!!
# # if element in mx1 or mx2 == 1, means that zombies can not go through this zone horizontal(mx1) or vertical(mx2)
# mx1 = np.zeros(g_map.shape)
# mx2 = np.zeros(g_map.shape)
# mx1[:,:] = 24
# mx2[:,:] = 24
# # block zone horizental
# for i in range(g_map.shape[1] - 10):
#     mx1[:, i] = np.all(g_map[:, i:10 + i] == wds1, axis=1)
# x_line, y_line = np.where(mx1)
# block_h = []
# for i in range(len(x_line)):
#     block_h.append((x_line[i], y_line[i]))
#
# # block zone vertical
# for i in range(g_map.shape[0] - 10):
#     mx2[i, :] = np.all(g_map[i:i + 10, :] == wds2, axis=0)
# x_col, y_col = np.where(mx2)
# block_v = []
# for i in range(len(x_col)):
#     block_v.append((x_col[i], y_col[i]))
# print('There are', len(block_h), 'zones where zombie can walk horizontally ')
# print('      and ', len(block_v), 'zones where zombie can walk vertically ')


def nbrs(now, g_map):
    """
    Function to find neighbors
    input now : tuple of index of the actual position (x,y)
        g_map: 2d np-array,in which we need to find the neighbors pixels of this pixels ---> g_map[now]
    output: surround: this is a list of neighbors ,each element is the coordinates of neighbor(x,y)
    """
    surround = []
    if now not in forbidden_zone:
        temp = [(now[0], now[1] - 1), (now[0] - 1, now[1]), (now[0] + 1, now[1]), (now[0], now[1] + 1)]
        for a in temp:
            if a[0] < 0 or a[1] < 0 or a[0] >= g_map.shape[0] or a[1] >= g_map.shape[1]:
                continue
            surround.append(a)
    shuffle(surround)
    return surround

def reachable(now,g_map,previous):
    """
        Juge the current pixel having weight 24 is reachable or not
    """
    for i in range(10):
        if now not in previous.keys():
            return True
        else:
            last = previous[now]
            if g_map[last] != 24:
                return True
        now = last
    return False

## Finding the shortest path algorithms begin from here
print('Dij: Calculate distance between start point and other point')
i = 0
j = 0
# when heap-min is empty , finish
while heap:
    # pop the minimum nodes
    thisWeight, thisNode = pop(heap)

    # Analyze the neighbors of current pixel
    for nbr in nbrs(thisNode, g_map):
        # Juge if this pixel is reachable
        if reachable(nbr, g_map, previous):
            # Here we iterate the 'distance'
                # Change the distance function means change Dij -> A*
            distance_nbr = thisWeight + g_map[nbr]

            # Renew the distance and previous if we have a better one
            if distances[nbr] > distance_nbr:
                distances[nbr] = distance_nbr
                push(heap, (distance_nbr, nbr))
                previous[nbr] = thisNode

        # if this pixel is not reachable ,store it
        else:
            forbidden_zone.append(nbr)

    ## statistic for the loop times.
    i += 1
    if i == 500000:
        i = 0
        j = j + 1
        print('-loop have run -', j, ' * 500K time', end='--/-- ')
        print('takes ', time.time() - b, 's', end='--/-- ')
        b = time.time()
        print('heapq length is ', len(heap))

print('Dijsktra: Distance takes ', time.time() - a, 's')
print("Unreachable pixel number ", len(forbidden_zone))
print('It will take ', distances[end], ' hours to arrive at ', (x_start, y_start))

#########################################################
## store the distance and previous step these two dictionaries in local as a pickle file
import pickle
print("Store the previous step into a file in order to reuse it in the test file")
output = open("D:\IMT_Atlantique\S4\ELU501\Challenge3\previous_zombie.pkl", "wb")
pickle.dump(previous, output)
output.close()
print("Store the distance into a file in order to reuse it in the test file")
output = open("D:\IMT_Atlantique\S4\ELU501\Challenge3\distance_zombie.pkl", "wb")
pickle.dump(distances, output)
output.close()
## Storage End



#########################################################
# Reform the path and draw the shortest path in image
path = []
now = end
np_img = np.int32(np.array(img))
while now != start:
    pre = previous[now]
    path.append(pre)
    now = pre
print('length of the path is', len(path))

for i in range(len(path)):
    np_img[path[i][0], path[i][1]] = (255, 0, 0)


# Show whole image with the shortest path in 2000 dpi
drawImg(np_img, 3000,"zombie_path_Rize2Brest.png")

# The time for arriving at Brest is 8842h if considering the propagation speed
#   depending on the population of humain linear.
# That means it will take 368days and 10h to come to Brest!!!!!!!!

