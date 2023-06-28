import os
import math
import h5py
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from math import radians, cos, sin, asin, sqrt
from mpl_toolkits.mplot3d import Axes3D
import bottom.get_bottom_surface as gbs
import bottom.setting_data.setting as st
upper, median, lower=gbs.get_gbs()

filepath,flight_path,strength,range_low,range_high,interval_x,interval_z=st.getsetting()
filename=os.path.split(filepath)[1]
beam=str(flight_path)+strength
jycs=0.1
in_range_distance=2.16
sjcs=3

def geodistance(lng1,lat1,lng2,lat2):
    # lng1,lat1,lng2,lat2 = (120.12802999999997,30.28708,115.86572000000001,28.7427)
    lng1, lat1, lng2, lat2 = map(radians, [float(lng1), float(lat1), float(lng2), float(lat2)])  # 经纬度转换成弧度
    dlon = lng2 - lng1
    dlat = lat2 - lat1
    a = sin(dlat / 2) ** 2 + cos(lat1) * cos(lat2) * sin(dlon / 2) ** 2
    distance = 2 * asin(sqrt(a)) * 6371 * 1000  # 地球平均半径，6371km
    # distance = distance / 1000000
    return distance
# aaa=geodistance(120,0,121,0)

# 公式6
def dist(x1,x2,y1,y2,a,b):
    dist=(((x1-x2)**2)/(a**2)+((y1-y2)**2)/(b**2))**0.5
    return dist

def dist_(x1,x2,y1,y2,a,b):
    dist=(((x1-x2)**2)/(a**2)+((y1-y2)**2)/(b**2))**0.5
    return dist

# 公式7
def cal_b0(hmax,hmin):
    b0=abs(hmax-hmin)
    return b0

def cal_rab(distance,d_datasat_z):
    Rab_sum=0

    # 统计柱状图曲线
    along_travel_interval=0.7
    m_d=np.max(distance)
    arr_along_travel = np.arange(0, m_d, along_travel_interval)
    # arr_along_travel
    arr_max_distance,arr_min_distance=[],[]
    try:
        count_fk = 0
        for i in range(len(arr_along_travel) - 1):
            left, right = arr_along_travel[i], arr_along_travel[i + 1]
            part_dis = []
            for j in range(len(distance)):
                if (distance[j] > left and distance[j] < right):
                    part_dis.append(d_datasat_z[j])
            if part_dis:
                count_fk=count_fk+1
                arr_max_distance.append(np.max(part_dis))
                arr_min_distance.append(np.min(part_dis))
    except:
        print(left,right)

    for i in range(len(arr_max_distance)):
        Rab_sum=Rab_sum+(0.7/(arr_max_distance[i]-arr_min_distance[i]))
    Rab=Rab_sum/len(arr_max_distance)
    return Rab

def cal_a0(b0,Rab):
    a0=b0*Rab
    return a0

# 公式8
def cal_min_pt0(d_datasat_x,d_datasat_y):
    arr_min_pt0=[]
    d_datasat_x_copy=d_datasat_x
    d_datasat_y_copy = d_datasat_y
    for i in range(len(d_datasat_x)):
        min_pt0=0
        for j in range(len(d_datasat_x)):
            distance=dist(d_datasat_x[i],d_datasat_x_copy[j],d_datasat_y[i],d_datasat_y_copy[j],a0,b0)
            if distance<in_range_distance:
                min_pt0 = min_pt0+1
        arr_min_pt0.append(min_pt0)
    ave_min_pt0=np.average(arr_min_pt0)
    return ave_min_pt0

# 公式9
def cal_bl(b0,WHl,Hmin):
    bl=b0+jycs*abs(WHl-Hmin)
    return bl

def cal_min_ptl(min_pt0,WHl,Hmin):
    min_ptl=min_pt0-int(abs(WHl-Hmin)/sjcs)
    return min_ptl

with h5py.File(filepath, mode='r') as f:
    # The gtl1 group means Ground Track L1.
    # latvar = f['/gt1l/geolocation/reference_photon_lat']

    latvar = f['/gt'+beam+'/heights/lat_ph']
    latitude = latvar[:]
    lat_vr = [latvar.attrs['valid_min'], latvar.attrs['valid_max']]

    # lonvar = f['/gt1l/geolocation/reference_photon_lon']
    lonvar = f['/gt'+beam+'/heights/lon_ph']
    longitude = lonvar[:]
    lon_vr = [lonvar.attrs['valid_min'], lonvar.attrs['valid_max']]

    # We'll plot h.
    h_var = f['/gt'+beam+'/heights/h_ph']
    temp = h_var[:]
    units = h_var.attrs['units']
    units = units.decode('ascii', 'replace')
    long_name = h_var.attrs['long_name']
    long_name = long_name.decode('ascii', 'replace')

    #画二维散点
    x, y, z =[],[],[]
    quality = f['/gt'+str(flight_path)+strength+'/heights/quality_ph']
    quality0 = quality[:]
    confidence = f['/gt'+str(flight_path)+strength+'/heights/signal_conf_ph']
    confidence4 = confidence[:]
    show_range_lat_min=range_low
    show_range_lat_max = range_high

    for i in range(len(temp)):
        if(latitude[i]>show_range_lat_min and latitude[i] < show_range_lat_max):
            x.append(latitude[i])
            y.append(longitude[i])
            z.append(temp[i])


    range_lat_min = min(x)
    range_lon_min = y[np.argmin(x)]
    range_lat_max = max(x)
    range_lon_max = y[np.argmax(x)]


    # print(np.argmin[x],np.argmax[x])


    def latlon_distance(x,y,z,lon_range_min, lat_range_min):
        distance = []
        for j in range(len(z)):
            distance_xy = geodistance(y[j], x[j], lon_range_min, lat_range_min)
            distance.append(distance_xy)
        return distance

    # 高斯去噪
    def select_gs(mu,sigma,x,y,z):
        distance_sealevel = []
        z_sealevel = []
        x_sealevel = []
        y_sealevel = []
        for i in range(len(z)):
            if (z[i] > mu - 3 * sigma and z[i] < mu + 3 * sigma):
                z_sealevel.append(z[i])
                x_sealevel.append(x[i])
                y_sealevel.append(y[i])
        return x_sealevel,y_sealevel,z_sealevel

    # 迭代高斯模型
    def multi_gs(gs_zc):
        gs_z = np.array(gs_zc)
        mu = np.mean(gs_z)  # 计算均值
        sigma = np.std(gs_z)
        return mu,sigma

    x_surface, y_surface, z_surface = [], [], []
    x_bottom, y_bottom, z_bottom = [], [], []
    for i in range(len(z)):
        if(z[i]>3.75 and z[i]<5.0):
            x_surface.append(x[i])
            y_surface.append(y[i])
            z_surface.append(z[i])
        elif(z[i]<lower):
            x_bottom.append(x[i])
            y_bottom.append(y[i])
            z_bottom.append(z[i])

    distance = latlon_distance(x, y, z,range_lon_min,range_lat_min)

    gs_z_surface = np.array(z_surface)
    mu_surface, sigma_surface = multi_gs(gs_z_surface)
    # print(mu_surface, sigma_surface)
    # x_surface_gs,y_surface_gs, z_surface_gs = select_gs(mu_surface, sigma_surface,x_surface,y_surface, z_surface)
    x_surface_gs, y_surface_gs, z_surface_gs=x_surface, y_surface, z_surface
    distance_surface_gs = latlon_distance(x_surface_gs, y_surface_gs, z_surface_gs,range_lon_min,range_lat_min)

    # 开始算法
    Hmax=np.max(z_surface_gs)
    Hmin = np.min(z_surface_gs)
    b0=abs(Hmax-Hmin)
    # Rab=cal_rab(distance_surface_gs, z_surface_gs)
    # Rab=upper-lower
    Rab=0.72
    a0=cal_a0(b0,Rab)

    min_pt0=cal_min_pt0(distance_surface_gs,z_surface_gs)
    x_lb,y_lb,z_lb=[],[],[]
    print("b0,Rab,a0,min_pt0:",b0,Rab,a0,min_pt0)
    for i in range(len(z_bottom)):
        WHl=z_bottom[i]
        bl = cal_bl(b0, WHl, Hmin)
        al = cal_a0(bl, Rab)
        min_ptl = cal_min_ptl(min_pt0, WHl, Hmin)
        # min_ptl = cal_min_ptl(min_pt0, WHl, Hmin)
        # try:
        #
        # except:
        #     print(b0)

        count=0

        distance_bottom=latlon_distance(x_bottom,y_bottom,z_bottom,range_lon_min,range_lat_min)
        distance_bottom_copy=distance_bottom
        z_bottom_copy=z_bottom
        for j in range(len(z_bottom)):
            dist = dist_(z_bottom[i],z_bottom_copy[j],distance_bottom[i],distance_bottom_copy[j],al,bl)
            if(dist<in_range_distance):
                count = count+1
        print(count)
        if(count>min_ptl):
            x_lb.append(x_bottom[i])
            y_lb.append(y_bottom[i])
            z_lb.append(z_bottom[i])
    distance_lb=latlon_distance(x_lb,y_lb,z_lb,range_lon_min,range_lat_min)
    plt.scatter(distance, z, s=0.4,c="b")
    plt.scatter(distance_surface_gs,z_surface_gs,s=0.4,c="g")
    plt.scatter(distance_lb, z_lb, s=0.4,c="r")
    plt.show()

    bottom_txt_name = filename + "_" + beam + "_" + str(round(range_low, 3)) + "_" + str(round(range_high, 3))
    filename_ = filename.split(".")[0]
    result_path = r'C:\\Users\\17826\\Desktop\Program\\bathymetry\\bottom\\result\\result_demo_12\\AVEBM\\' + bottom_txt_name + ".txt"


    # 头文件字符串
    def write_con(x, y, z):
        with open(result_path, "a", encoding='utf-8') as f:
            for i in range(len(x)):
                f.write(str(x[i]) + " " + str(y[i]) + " " + str(z[i]) + '\n')


    # # 生成的数据写入到文件
    with open(result_path, "a", encoding='utf-8') as f:
        f.write("lat lon elevation" + '\n')
    write_con(x_lb, y_lb, z_lb)
