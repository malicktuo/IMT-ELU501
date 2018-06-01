import PIL as im
import numpy as np
from matplotlib import pyplot as plt
import time
from heapq import heappop as pop
from heapq import heappush as push
from random import shuffle
from math import radians, cos, sin, asin, sqrt

def haversine(lat1, lat2, lon1,lon2):
    """
    Calculate the great circle distance between two points
    on the earth (specified in decimal degrees)
    """
    lat1, lat2,lon1,lon2 = map(radians, [lat1, lat2,lon1,lon2])

    # haversine equation
    dlon1 = lon2-lon1
    dlat1 = lat2 - lat1
    a1 = sin(dlat1 / 2) ** 2 + cos(lat1) * cos(lat2) * sin(dlon1 / 2) ** 2
    c1 = 2 * asin(sqrt(a1))
    r1 = 6371  #  radium of earth
    return c1 * r1

def deployTemp(temp_one_day):
    """

    :return:
    """
    (x,y) = temp_one_day.shape
    lat_pixel = {}
    for i in range(x):
        lat1 = temp_one_day[i,0]
        lon1 = temp_one_day[i,1]
        del_x = haversine(lat1, lat1-1,lon1,lon1)
        del_y = haversine(lat1, lat1, lon1, lon1-1)
        lat_pixel[lat1] = np.ceil((del_x,del_y))
    return lat_pixel


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
            tem_img: ndarray,365 images with useful temperature information

    """

    # generate 365 images with temperature information
    pre_t = time.time()
    file_name = "Temperature_Rize2Brest.npy"
    f = open(file_name, 'rb')

    # temperature[a,b,c]
    # a: day
    # b: location
    # c: attribute(0:latidude   1:longitude  2:temperature in degree C in range of (0,18))
    temperature = np.load(f)
    print("Reading temperature takes ", time.time() - pre_t, " s")
    print("data shape ", temperature.shape)
    f.close()

    ######################################################################################
    m = deployTemp(temperature[0, :, :])

    # Reduce the scale of Image
    x1 = 1280
    x2 = 2200
    y1 = 600
    y2 = 4450
    brest = (1306 - x1, 669 - y1)
    rize = (2108 - x1, 4426 - y1)
    (x, y, z) = temperature.shape

    tmp_img = np.zeros((x, x2 - x1, y2 - y1))
    for day in range(x):
        x_t = 0
        y_t = 0
        index = 0
        for line in range(y):
            lat, longi, t = temperature[day, line, :]
            del_x, del_y = m[lat]
            tmp_img[day, int(x_t):int(np.round(x_t + del_x)), int(y_t):int(np.round(y_t + del_y+10))] = t
            y_t = y_t + del_y+10
            index += 1
            if index % 46 == 0:  # 45 longitude slots
                # print(lat,longi,t)
                index = 0
                x_t = x_t + del_x
                y_t = 0

    # read image RGB from this directory
    input_filename = "D:\IMT_Atlantique\S4\ELU501\Challenge3\population-density-map.bmp"
    img = im.Image.open(input_filename)

    # generate a numpy array to represent image
    G_orig = np.array(img)

    # Reduce the scale of Image

    G_orig = G_orig[x1:x2,y1:y2]

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
    G_gray = G_gray[x1:x2,y1:y2]

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

    return g_normalised, startPoint[0][0], startPoint[1][0], endPoint[0][0], endPoint[1][0], G_orig, tmp_img


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
g_map, x_start, y_start, x_end, y_end, img, tmp_img = preprocessing()


print('-- In New Graph --- ')
print('startPoint is ', (x_start, y_start))
print('endPoint is ', (x_end, y_end))
print('temperature shape',tmp_img.shape)
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
            and also if the temperature will be below 0 for a week
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

def reachable_temperature(now,tmp_img,distance):
    """
        Juge the current pixel is reachable if the temperature will be higher than 0 at least one day in the next week
    """
    if distance > 999999999:  # new pixel.
        return True
    day = int(np.round(distance / 24) % 365)
    for i in range(7):
        t = tmp_img[day+i,now[0],now[1]]
        if t>0:
            return True
    return False

def time_with_temperature(time_org,tmp_img,now,distance):
    """
    A star algo, function to calculate the cost.
                    will return the period cause temperature influence the speed
    """
    if distance > 999999999:  # new pixel.
        t = tmp_img[0,now[0],now[1]]
    else:  # calculate which day  in order to find the proper temperature
        day = int(np.round(distance / 24) % 365)
        t = tmp_img[day,now[0],now[1]]

    # Calculate when temperature =< 0, zombie will be still , when t >=18 zombie move as usual
    old_speed = 1/time_org
    new_speed = old_speed * (t/18)
    return (1/new_speed)

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
        if reachable(nbr, g_map,previous) and reachable_temperature(nbr,tmp_img,distances[nbr]):
            # Here we iterate the 'distance'
                # Change the distance function means change Dij -> A*
            distance_nbr = thisWeight + time_with_temperature(g_map[nbr],tmp_img,nbr,thisWeight)

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
output = open("D:\IMT_Atlantique\S4\ELU501\Challenge3\previous_zombie_with_t.pkl", "wb")
pickle.dump(previous, output)
output.close()
print("Store the distance into a file in order to reuse it in the test file")
output = open("D:\IMT_Atlantique\S4\ELU501\Challenge3\distance_zombie_with_t.pkl", "wb")
pickle.dump(distances, output)
output.close()
## Storage End