import h5py
import matplotlib
import matplotlib.pyplot as plt
from math import radians, cos, sin, asin, sqrt
import bottom.setting_data.setting as st
import os.path
import numpy as np

filepath,flight_path,strength,range_low,range_high,interval_x,interval_z=st.getsetting()
filename=os.path.split(filepath)[1]
beam=str(flight_path)+strength
show_range_lat_min=range_low
show_range_lat_max=range_high


# filepath1=r"E:\Data\IceSat-2\nanhai_atl03\night\ATL03_20210818183003_08571207_005_01.h5"
x,y,z=[],[],[]
sub_x,sub_y,sub_z=[],[],[]

op_x,op_y,op_z=[],[],[]

with h5py.File(filepath, mode='r') as f:
    # The gtl1 group means Ground Track L1.
    # latvar = f['/gt1l/geolocation/reference_photon_lat']

    latvar = f['/gt'+str(flight_path)+strength+'/heights/lat_ph']
    latitude = latvar[:]
    lat_vr = [latvar.attrs['valid_min'], latvar.attrs['valid_max']]

    # lonvar = f['/gt1l/geolocation/reference_photon_lon']
    lonvar = f['/gt'+str(flight_path)+strength+'/heights/lon_ph']
    longitude = lonvar[:]
    lon_vr = [lonvar.attrs['valid_min'], lonvar.attrs['valid_max']]

    # We'll plot h.
    h_var = f['/gt'+str(flight_path)+strength+'/heights/h_ph']
    temp = h_var[:]
    # units = h_var.attrs['units']
    # units = units.decode('ascii', 'replace')
    # long_name = h_var.attrs['long_name']
    # long_name = long_name.decode('ascii', 'replace')

    #画三维散点
    # x, y, z =[],[],[]
    quality = f['/gt'+str(flight_path)+strength+'/heights/quality_ph']
    quality0 = quality[:]
    confidence = f['/gt'+str(flight_path)+strength+'/heights/signal_conf_ph']
    confidence4 = confidence[:]
    # 033-073 225-255

    for i in range(len(temp)):
        if(latitude[i]>range_low and latitude[i] < range_high):
            x.append(latitude[i])
            y.append(longitude[i])
            z.append(temp[i])
            # print(x,y)
            # if (quality0[i] == 0):
            #     if (confidence4[i][0] ==4):
            #         x.append(latitude[i])
            #         y.append(longitude[i])
            #         z.append(temp[i])

    for i in range(len(x)):
        if(z[i]<1.5):
            sub_x.append(x[i])
            sub_y.append(y[i])
            sub_z.append(z[i])

            # print(x,y)
            # if (quality0[i] == 0):
            #     if (confidence4[i][0] ==4):
            #         x.append(latitude[i])
            #         y.append(longitude[i])
            #         z.append(temp[i])

#3经纬度wgs84转utm
def wgs84_utm(lat,lon):
    from pyproj import  Transformer
    transformer=Transformer.from_crs("epsg:4326", "epsg:32649")
    utm_x, utm_y = transformer.transform(lat,lon)
    return utm_x,utm_y



# 4.OPTICS方法去噪,输入utm坐标系下的纬度和高程，返回cluster_1,该numpy第一个参数是去噪后的纬度，第二个参数是去噪后的经度
def my_optics(x,y,z):
    from sklearn.cluster import OPTICS
    utm_x, utm_y = wgs84_utm(x, y)
    X=np.dstack((utm_x,z)).reshape(-1,2)
    clustering = OPTICS(min_samples=20).fit(X)
    labels=clustering.labels_

    for i in range(len(labels)):
        if(labels[i]!=-1):
            op_x.append(x[i])
            op_y.append(y[i])
            op_z.append(z[i])

    return op_x,op_y,op_z

my_optics(sub_x,sub_y,sub_z)#  传入纬度，高程，经度

range_lat_min = min(x)
range_lon_min = y[np.argmin(x)]
range_lat_max = max(x)
range_lon_max = y[np.argmax(x)]
# print(np.argmin[x],np.argmax[x])

print("range_lat_min:", range_lat_min, "range_lon_min:", range_lon_min,
      "range_lat_max:", range_lat_max, "range_lon_max:", range_lon_max)


def geodistance(lng1, lat1, lng2, lat2):
    # lng1,lat1,lng2,lat2 = (120.12802999999997,30.28708,115.86572000000001,28.7427)
    lng1, lat1, lng2, lat2 = map(radians, [float(lng1), float(lat1), float(lng2), float(lat2)])  # 经纬度转换成弧度
    dlon = lng1 - lng2
    dlat = lat1 - lat2
    a = sin(dlat / 2) ** 2 + cos(lat1) * cos(lat2) * sin(dlon / 2) ** 2
    distance = 2 * asin(sqrt(a)) * 6371 * 1000  # 地球平均半径，6371km
    distance = round(distance, 2)
    return distance


def latlon_distance(x, y, z):
    distance = []
    for j in range(len(z)):
        distance_xy = geodistance(y[j], x[j], range_lon_min, range_lat_min)
        distance.append(distance_xy)
    return distance


# 绘制二维散点图
distance = latlon_distance(x, y, z)
distance_signal = latlon_distance(op_x,op_y,op_z)


plt.scatter(distance, z, s=10, c="#5872f9", label='Raw Photons', zorder=1)
plt.scatter(distance_signal, op_z, s=10, c="#FF0000", label='OPTICS Photons', zorder=1)
# 横纵标题格式
font2 = {'family': 'Times New Roman',
         'weight': 'bold',
         'size': 40,
         }
font3 = {'family': 'Times New Roman',
         'weight': 'normal',
         'size': 30,
         }
# 刻度标注大小
plt.xticks(fontproperties='Times New Roman', fontsize=30)
plt.yticks(fontproperties='Times New Roman', fontsize=30)

plt.legend(loc="upper right", prop=font3, ncol=2)
# 无空白
plt.margins(0, 0)


plt.xlabel("Along Track, m", font2)
plt.ylabel("Elevation, m", font2)

plt.subplots_adjust(left=0.088, bottom=0.147, right=0.985, top=0.983, wspace=0.2, hspace=0.2)

plt.show()

bottom_txt_name = filename + "_" + beam + "_" + str(round(range_low, 3)) + "_" + str(round(range_high, 3))
filename_ = filename.split(".")[0]
result_path = r'C:\\Users\\17826\\Desktop\Program\\bathymetry\\bottom\\result\\result_demo_12\\OPTICS\\' + bottom_txt_name + ".txt"


# 头文件字符串
def write_con(x, y, z):
    with open(result_path, "a", encoding='utf-8') as f:
        for i in range(len(x)):
            f.write(str(x[i]) + " " + str(y[i]) + " " + str(z[i]) + '\n')


# # 生成的数据写入到文件
with open(result_path, "a", encoding='utf-8') as f:
    f.write("lat lon elevation" + '\n')
write_con(op_x,op_y,op_z)