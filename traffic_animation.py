import datetime
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from matplotlib.animation import FuncAnimation

info = pd.read_csv('STGCN_IJCAI-18-master/dataset/PeMSD7_M_Station_Info.csv')
# print(info.head())

longitude_min, longitude_max = info['Longitude'].min(), info['Longitude'].max()
latitude_min, latitude_max = info['Latitude'].min(), info['Latitude'].max()
# add % buffer to each side
longitude_range = longitude_max - longitude_min
latitude_range = latitude_max - latitude_min
longitude_buffer_percent = 0.05
latitude_buffer_percent = 0.05
longitude_buffer = longitude_range * longitude_buffer_percent
latitude_buffer = latitude_range * latitude_buffer_percent
longitude_min -= longitude_buffer
longitude_max += longitude_buffer
latitude_min -= latitude_buffer
latitude_max += latitude_buffer
# floor/ceil to 2 decimal places
longitude_min = np.floor(longitude_min * 100) / 100
longitude_max = np.ceil(longitude_max * 100) / 100
latitude_min = np.floor(latitude_min * 100) / 100
latitude_max = np.ceil(latitude_max * 100) / 100
print(longitude_min, longitude_max)
print(latitude_min, latitude_max)

V = pd.read_csv('STGCN_IJCAI-18-master/dataset/PeMSD7_Full/PeMSD7_V_228.csv', header=None).values
print(V.shape)

seconds_per_day = 12
# each frame is 5 minutes
days_per_frame = 5 / 60 / 24
fps = 1 / (seconds_per_day * days_per_frame)
frame_count = 288 * 3 # len(V)
scale = 3

fig = plt.figure()
# plot the map
img = mpimg.imread(f'map_{longitude_buffer_percent}_{latitude_buffer_percent}.png')
# avg color channels
img = img[:,:,:3].mean(axis=2)
# change x y limits to match the map
plt.xlim(longitude_min, longitude_max)
plt.ylim(latitude_min, latitude_max)
# plot the map in the limits
plt.imshow(img, extent=[longitude_min, longitude_max, latitude_min, latitude_max], cmap='gray')
# plot all of the stations
print()
scat = plt.scatter(info['Longitude'], info['Latitude'], s=50, c=V[0], cmap='RdYlGn', edgecolor='black', vmin=0, vmax=V.max())

cbar = plt.colorbar(fraction=0.025)
cbar.ax.get_yaxis().labelpad = 15
cbar.set_label("Average Speed (mph)",rotation=270)

plt.xlabel('Longitude')
plt.ylabel('Latitude')

def frame_to_time(frame_number):
    datapoint = frame_number
    weekdays = datapoint // 288
    minutes = (datapoint % 288) * 5
    weekends = weekdays // 5
    days = weekdays + weekends * 2
    dt = datetime.datetime(2012, 5, 1) + datetime.timedelta(days=days, minutes=minutes)
    return dt.strftime("%A %B %d, %Y %I:%M %p")

def update(frame_number):
    # scat.set_sizes(V[frame_number] * scale)
    scat.set_array(V[frame_number])
    plt.title(frame_to_time(frame_number))
ani = FuncAnimation(fig, update, interval=1000/fps, save_count=frame_count)
# ani.save('animation.mp4')
plt.show()