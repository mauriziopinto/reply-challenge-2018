import numpy as np
import scipy.misc
import datetime

from skimage.draw import polygon
from skimage.draw import polygon_perimeter
from skimage.draw import ellipse
from heapq import *

f_name = "input_3.txt"

# load start and end positions
with open(f_name, 'r') as f:
	start_end = np.fromstring(f.readline(), dtype=float, sep=" ")

start_x = start_end[0]
start_y = start_end[1]
end_x = start_end[2]
end_y = start_end[3]

print("Original start and end: (", start_x, ",", start_y, ") (", end_x, ",", end_y, ")")

# load obstacles, all lines skipping first 2
data = np.loadtxt(f_name, delimiter=" ", skiprows=2)

#[:,n] takes all the values in column n of the input matrix
x = np.stack( (data[:,0], data[:,2], data[:,4]), axis=0)
y = np.stack( (data[:,1], data[:,3], data[:,5]), axis=0)

min_x = min(np.min(x),start_x,end_x)
max_x = max(np.max(x),start_x,end_x)

min_y = min(np.min(y),start_y,end_y)
max_y = max(np.max(y),start_y,end_y)

print("Original window: ", min_x, min_y, max_x, max_y)

shift_x = 0 - min_x
shift_y = 0 - min_y

data[:,0] = data[:,0] + shift_x
data[:,2] = data[:,2] + shift_x
data[:,4] = data[:,4] + shift_x

data[:,1] = data[:,1] + shift_y
data[:,3] = data[:,3] + shift_y
data[:,5] = data[:,5] + shift_y

x = np.stack( (data[:,0], data[:,2], data[:,4]), axis=0)
y = np.stack( (data[:,1], data[:,3], data[:,5]), axis=0)

start_x = int(start_x + shift_x)
start_y = int(start_y + shift_y)
end_x = int(end_x + shift_x)
end_y = int(end_y + shift_y)

min_x = min(np.min(x),start_x,end_x)
max_x = max(np.max(x),start_x,end_x)

min_y = min(np.min(y),start_y,end_y)
max_y = max(np.max(y),start_y,end_y)

print("Shifted window: ", min_x, min_y, max_x, max_y)


print("Shifted start and end: (", start_x, ",", start_y, ") (", end_x, ",", end_y, ")")

width = int(max_x) +1
height = int(max_y) +1
print("Blank matrix created dimensions: (", width, ",", height, ")") 
# create a blank image (all zeroes) / map
img = np.zeros((width, height), dtype=np.uint8)
print("Shape ",np.shape(img))

now = datetime.datetime.now()
print("loading Obstacles  ",now.hour,":", now.minute)

# draw obstacles
for obstacle in data: #[:5000]:
	x1 = int(obstacle[0])
	y1 = int(obstacle[1])
	x2 = int(obstacle[2])
	y2 = int(obstacle[3])
	x3 = int(obstacle[4])
	y3 = int(obstacle[5])
	r = np.array([x1, x2, x3])
	c = np.array([y1, y2, y3])
	rr, cc = polygon(r, c)
	#rr, cc = polygon_perimeter(r, c)
	#print("Obstacle n:",rr, cc)
	img[rr, cc] = 1

now = datetime.datetime.now()
print("Obstacles loaded",now.hour,":", now.minute)

# mark start and end (test only)
#rr, cc = ellipse(start_x, start_y, 30, 30)
#img[rr, cc] = 255

#rr, cc = ellipse(end_x, end_y, 30, 30)
#img[rr, cc] = 255

# a*
import heapq

class PriorityQueue:
    def __init__(self):
        self.elements = []
    
    def empty(self):
        return len(self.elements) == 0
    
    def put(self, item, priority):
        heapq.heappush(self.elements, (priority, item))
    
    def get(self):
        return heapq.heappop(self.elements)[1]

def heuristic(a, b):
	D=1
	D2=1
	dx = abs(a[0] - b[0])
	dy = abs(a[1] - b[1])
	return D * (dx + dy) + (D2 - 2 * D) * min(dx, dy)


def heuristic2(a, b):
	return (b[0] - a[0]) ** 2 + (b[1] - a[1]) ** 2

def astar(array, start, goal):

	neighbors = [(0,1),(0,-1),(1,0),(-1,0),(1,1),(1,-1),(-1,1),(-1,-1)]

	close_set = set()
	came_from = {}
	gscore = {start:0}
	fscore = {start:heuristic(start, goal)}
	oheap = []

	heappush(oheap, (fscore[start], start))
	
	while oheap:
        
		current = heappop(oheap)[1]

		if current == goal:
			data = []
			while current in came_from:
				data.append(current)
				current = came_from[current]
			return data

		close_set.add(current)
		for i, j in neighbors:
			neighbor = current[0] + i, current[1] + j            
			tentative_g_score = gscore[current] + heuristic(current, neighbor)
			#print("Punti (",neighbor[0],",",neighbor[1], ")")
			#print("Punto=",array[neighbor[0]][neighbor[1]])

			if 0 <= neighbor[0] < array.shape[0]:
				if 0 <= neighbor[1] < array.shape[1]:
					if array[neighbor[0]][neighbor[1]] == 1:
						continue
				else:
					# array bound y walls
					continue
			else:
				# array bound x walls
				continue
				
			if neighbor in close_set and tentative_g_score >= gscore.get(neighbor, 0):
				continue
				
			if  tentative_g_score < gscore.get(neighbor, 0) or neighbor not in [i[1]for i in oheap]:
				came_from[neighbor] = current
				gscore[neighbor] = tentative_g_score
				fscore[neighbor] = tentative_g_score + heuristic(neighbor, goal)
				heappush(oheap, (fscore[neighbor], neighbor))
				
	return False

'''Here is an example of using my algo with a numpy array,
   astar(array, start, destination)
   astar function returns a list of points (shortest path)'''

nmap = np.array(img)
print(nmap)
now = datetime.datetime.now()
print("Path finding started...",now.hour,":", now.minute)

path = astar(nmap, (start_x, start_y), (end_x, end_y))
now = datetime.datetime.now()
print("Path finding completed!",now.hour,":", now.minute)

#img[img > 0] = 255

print("Shifted path ",path)
#rgb_img = np.stack((img,)*3, -1)

#for step in path:
#	rr, cc = ellipse(step[0], step[1], 1, 1)
#	rgb_img[rr, cc, 0] = 255
#	rgb_img[rr, cc, 1] = 0
#	rgb_img[rr, cc, 2] = 0

# resize and save as image
#resized = scipy.misc.imresize(rgb_img, (int(max_x/5), int(max_y/5)))
#scipy.misc.imsave("map.png", rgb_img)

# TODO shift the coordinates to the original reference (-shiftx, -shifty)
# TODO save the path elements to file


print("Completed")
