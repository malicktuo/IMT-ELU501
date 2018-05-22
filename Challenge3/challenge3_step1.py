import PIL as im
import numpy as np
from matplotlib import pyplot as plt
import time
from heapq import heappop as pop
from heapq import heappush as push
from random import shuffle


def findIndexInMap(g_map, val):
    mask = np.all(g_map.reshape(-1, 1) == val, axis=1)
    b = np.where(mask.reshape(g_map.shape))
    print(g_map[b[0], b[1]])
    return b[0], b[1]


pre_pro = time.time()
input_filename = "D:\IMT_Atlantique\S4\ELU501\Challenge3\population-density-map.bmp"
img = im.Image.open(input_filename)

G_orig = np.array(img)
# Find start(green) and end(red) point
# start:G_orig[2108,4426]   G_orig[startPoint[0],startPoint[1]]
# end: G_orig[1306,669]    G_orig[endPoint[0],endPoint[1]]
mask = np.all(G_orig == (255, 0, 0), axis=-1)
endPoint = np.where(mask)
mask = np.all(G_orig == (0, 255, 0), axis=-1)
startPoint = np.where(mask)

# change to gray
img_gray = img.convert("L")
G_gray = np.float64((np.array(img_gray)))

g_normalised = (G_gray - (np.min(G_gray))) / (np.max(G_gray) - np.min(G_gray))
print(np.unique(g_normalised, return_counts=True))
g_normalised = -23 * g_normalised + 24
g_normalised = np.int32(np.round(g_normalised))
print(np.unique(g_normalised, return_counts=True))

# create map :g_map , values in the map means hours to pass this pixel
#
# x_start = startPoint[0][0]
# y_start = startPoint[1][0]


#### Paris (1250,1200)
#### Londre (1000,1000)
#### Renne (1337,888)
# start:G_orig[2108,4426]   G_orig[startPoint[0],startPoint[1]]
# end: G_orig[1306,669]    G_orig[endPoint[0],endPoint[1]]
# interval [1000:2500,500:4500]
x_start = 2108
y_start = 4426
xval = 1200
yval = 600
val = 1000
g_map = g_normalised
# g_map = g_normalised[xval:xval + val, yval:yval + val]
#
# x_end = endPoint[0][0] - xval
# x_start = x_start - xval
# y_end = endPoint[1][0] - yval
# y_start = y_start - yval
x_start,y_start = x_start,y_start
x_end,y_end = endPoint[0][0],endPoint[1][0]
print('--in new graphe--- ')
print('startPoint is ', (x_start, y_start))
print('endPoint is ', (x_end, y_end))

# make 2 windows in order to find block zones
wds1 = np.zeros((1, 10))
wds2 = np.zeros((10, 1))
mx1 = np.zeros(g_map.shape)
mx2 = np.zeros(g_map.shape)
# block zone horizental
for i in range(g_map.shape[1] - 10):
    mx1[:, i] = np.all(g_map[:, i:10 + i] == wds1, axis=1)
x_line, y_line = np.where(mx1)
block_h = []
for i in range(len(x_line)):
    block_h.append((x_line[i], y_line[i]))

# block zone vertical
for i in range(g_map.shape[0] - 10):
    mx2[i, :] = np.all(g_map[i:i + 10, :] == wds2, axis=0)
x_col, y_col = np.where(mx2)
block_v = []
for i in range(len(x_col)):
    block_v.append((x_col[i], y_col[i]))


### dijkstra

def nbrs(now, g_map=g_map):
    surround = []
    if now in block_v:
        print('vertical block ')
        temp = [(now[0], now[1] - 1), (now[0], now[1] + 1)]
        for a in temp:
            if a[0] < 0 or a[1] < 0 or a[0] >= g_map.shape[0] or a[1] >= g_map.shape[1]:
                continue
            surround.append(a)
    elif now in block_h:
        print('horizontal block ')
        temp = [(now[0] - 1, now[1]), (now[0] + 1, now[1])]
        for a in temp:
            if a[0] < 0 or a[1] < 0 or a[0] >= g_map.shape[0] or a[1] >= g_map.shape[1]:
                continue
            surround.append(a)
    else:
        temp = [(now[0], now[1] - 1), (now[0] - 1, now[1]), (now[0] + 1, now[1]), (now[0], now[1] + 1)]
        for a in temp:
            if a[0] < 0 or a[1] < 0 or a[0] >= g_map.shape[0] or a[1] >= g_map.shape[1]:
                continue
            surround.append(a)
    shuffle(surround)
    return surround


start = (x_start, y_start)
end = (x_end, y_end)
visited = []
path = {}
now = start
shape = g_map.shape
heap = []
push(heap, (g_map[start], start))
previous = {}
distances = {}  # distance between start node and others  {end:distance}
lastPoint = {}

for x in range(shape[0]):
    for y in range(shape[1]):
        distances[(x, y)] = float('inf')
distances[start] = 0
a = time.time()
print('\npreprocessing takes ', a - pre_pro, 's \n')

b = a
print('Dij: Calculate distance between start point and other point')
i = 0
j = 0
while heap:
    thisWeight, thisNode = pop(heap)
    for nbr in nbrs(thisNode):
        distance_nbr = thisWeight + g_map[nbr]
        if distances[nbr] > distance_nbr:
            distances[nbr] = distance_nbr
            push(heap, (distance_nbr, nbr))
            previous[nbr] = thisNode

    ## statistic for the loop times.
    i += 1
    if i == 100000:
        i = 0
        j = j + 1
        print('-loop have run -', j, ' *100 K time', end='--/-- ')
        print('takes ', time.time() - b, 's', end='--/-- ')
        b = time.time()
        print('heapq length is ',len(heap))

#
# while now != end:
#     visited.append(now)
#     t = 999
#     for next in surround:
#         if next[0] > 0 and next[1] > 0 and t < g_map[next] and next not in visited:
#             path[next] = now
#             now = next
#             t = g_map[next]

print('distance counting takes time ', time.time() - a, 's')

print(distances[end])

# find a path
path = []
now = end
print(distances[now])

while now != start:
    pre = previous[now]
    path.append(pre)
    now = pre

for i in range(75, 81):
    print([100, i])


np_img = np.int32(np.array(img))
#### Paris (1250,1200)
#### Londre (1000,1000)
#### Renne (1337,888)
# start:G_orig[2108,4426]   G_orig[startPoint[0],startPoint[1]]
# end: G_orig[1306,669]    G_orig[endPoint[0],endPoint[1]]
# interval
# np_img = np_img[1000:2500,500:4500]
# np_img = np_img[xval:xval + val, yval:yval + val]
for i in range(len(path)):
    np_img[path[i][0], path[i][1] - 1] = (255, 0, 0)
plt.figure(dpi=200)
plt.imshow(np_img)
plt.show()
