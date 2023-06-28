import os
import h5py
import numpy as np
import matplotlib.pyplot as plt
from math import radians, cos, sin, asin, sqrt
import bottom.setting_data.setting as st

filepath,flight_path,strength,range_low,range_high,interval_x,interval_z=st.getsetting()
filename=os.path.split(filepath)[1]
beam=str(flight_path)+strength
show_range_lat_min=range_low
show_range_lat_max=range_high
# FILE_NAME =r'E:\\Data\\IceSat-2\\nanhai_atl03\\day\\ATL03_20210214031837_07961007_005_01.h5'
with h5py.File(filepath, mode='r') as f:
    # The gtl1 group means Ground Track L1.
    # latvar = f['/gt1l/geolocation/reference_photon_lat']
    # flight_path = 1
    # strength = 'r'
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
    units = h_var.attrs['units']
    units = units.decode('ascii', 'replace')
    long_name = h_var.attrs['long_name']
    long_name = long_name.decode('ascii', 'replace')

    #画三维散点
    x, y, z,x_surface,y_surface,z_surface =[],[],[],[],[],[]
    quality = f['/gt'+str(flight_path)+strength+'/heights/quality_ph']
    quality0 = quality[:]
    confidence = f['/gt'+str(flight_path)+strength+'/heights/signal_conf_ph']
    confidence4 = confidence[:]
    # 033-073 225-255
    # show_range_lat_min=16.81
    # show_range_lat_max = 16.84
    # filename = os.path.split(filepath)[1]
    # range_txt_path = ".\\ss_result\\DSUM\\" + filename + ".txt"
    # range_txt_result=".\\ss_result\\"
    # method_result="DBSCAN\\"
    # range_txt_path = ".\\ss_result\\" +method_result+ filename + ".txt"

    interval_get_sur=interval_x
    count_interval_get_sur=int((show_range_lat_max-show_range_lat_min)/interval_get_sur)

    for i in range(len(temp)):
        if (latitude[i] > show_range_lat_min and latitude[i] < show_range_lat_max and temp[i]<3.8):
            x.append(latitude[i])
            y.append(longitude[i])
            z.append(temp[i])

    def DBSCAN(distance,z):
        distance_copy=distance
        distance_point=[]
        for i in range(len(distance_copy)):
            sum=0
            for j in range(len(distance)):
                dis=(distance_copy[i]-distance[j])**2+(z[i]-z[j])**2
                if(dis<1):
                    sum=sum+1
            distance_point.append(sum)
        for i in range(len(distance_point)):
            if(distance_point[i]>2):
                x_surface.append(x[i])
                y_surface.append(y[i])
                z_surface.append(z[i])
        return x_surface,y_surface,z_surface

    range_lat_min=min(x)
    range_lat_min_index=np.argmin(x)
    range_lon_min=y[range_lat_min_index]

    print("range_lat_min:",range_lat_min,"range_lon_min:",range_lon_min)


    def geodistance(lng1, lat1, lng2, lat2):
        # lng1,lat1,lng2,lat2 = (120.12802999999997,30.28708,115.86572000000001,28.7427)
        lng1, lat1, lng2, lat2 = map(radians, [float(lng1), float(lat1), float(lng2), float(lat2)])  # 经纬度转换成弧度
        dlon = lng1 - lng2
        dlat = lat1 - lat2
        a = sin(dlat / 2) ** 2 + cos(lat1) * cos(lat2) * sin(dlon / 2) ** 2
        distance = 2 * asin(sqrt(a)) * 6371 * 1000  # 地球平均半径，6371km
        distance = round(distance,2)
        return distance


    def latlon_distance(x, y, z):
        distance = []
        for j in range(len(z)):
            distance_xy = geodistance(y[j], x[j], range_lon_min, range_lat_min)
            distance.append(distance_xy)
        return distance

    def getevavaluexy(distance_surface,z_surface):
        # 统计每隔一段距离的中位数高程，这里是10m
        def get_median(data):
            data.sort()
            half = len(data) // 2
            return (data[half] + data[~half]) / 2

        interval_median = 10
        median_array_x_interbal = np.arange(0, max(distance_surface), interval_median)
        median_array_x = []
        for i in range(len(median_array_x_interbal) - 1):
            median_array_x.append(median_array_x_interbal[i] + interval_median / 2)

        list_median = []

        for i in range(len(median_array_x_interbal) - 1):
            range_low = median_array_x_interbal[i]
            range_high = median_array_x_interbal[i + 1]
            list_single = []
            for j in range(len(distance_surface)):
                if (distance_surface[j] > range_low and distance_surface[j] < range_high):
                    list_single.append(z_surface[j])

            if (len(list_single)!=0):
                list_median.append(get_median(list_single))
            else:
                list_median.append(35767)

        return  median_array_x,list_median

    #绘制二维散点图
    distance=latlon_distance(x,y,z)

    x_surface,y_surface,z_surface=DBSCAN(distance, z)


    distance_surface=latlon_distance(x_surface,y_surface,z_surface)

    plt.scatter(distance,z,s=0.4,label="all points")
    plt.scatter(distance_surface,z_surface,s=1,color="red",label="points of surface")


    # 点结果文件
    # range_txt_path_point = ".\\ss_result\\" +method_result+str(range_low)+"_"+str(range_high) +filename + ".txt"
    # f_point = open(range_txt_path_point, 'a')
    # for i in range(len(distance_surface)):
    #     new_context = str(distance_surface[i]) + " " + str(z_surface[i]) + '\n'
    #     f_point.writelines(new_context)

    # 10m区间
    # median_array_x, list_median=getevavaluexy(distance_surface,z_surface)
    # range_txt_path_10m = ".\\ss_result\\" +method_result+"10m_" +str(range_low)+"_"+str(range_high)+"_"+filename + ".txt"
    # f = open(range_txt_path_10m, 'a')
    # for tenm in range(len(median_array_x)):
    #     new_context = str(median_array_x[tenm]) + " " + str(list_median[tenm]) + '\n'
    #     f.writelines(new_context)

    median_array_x_unnon=[]
    list_median_unnon=[]
    # for i in range(len(list_median)):
    #     if(list_median[i]!=35767):
    #         median_array_x_unnon.append(median_array_x[i])
    #         list_median_unnon.append(list_median[i])
    # plt.plot(median_array_x_unnon, list_median_unnon, color="green", label="elevation-10m")


    #横纵标题格式
    font2 = {'family': 'Times New Roman',
             'weight': 'normal',
             'size': 20,
             }

    #刻度标注大小
    plt.xticks(fontsize=15)
    plt.yticks(fontsize=15)
    # 无空白
    plt.margins(0, 0)

    plt.xlabel("Along Track, m",font2)
    plt.ylabel("Elevation, m",font2)
    plt.legend(fontsize=15)
    plt.legend(fontsize=15,loc="upper left")
    plt.show()


    bottom_txt_name = filename + "_" + beam + "_" + str(round(range_low, 3)) + "_" + str(round(range_high, 3))
    filename_ = filename.split(".")[0]
    result_path = r'C:\\Users\\17826\\Desktop\Program\\bathymetry\\bottom\\result\\result_demo_12\\DBSCAN\\' + bottom_txt_name + ".txt"


    # 头文件字符串
    def write_con(x, y, z):
        with open(result_path, "a", encoding='utf-8') as f:
            for i in range(len(x)):
                f.write(str(x[i]) + " " + str(y[i]) + " " + str(z[i]) + '\n')


    # # 生成的数据写入到文件
    with open(result_path, "a", encoding='utf-8') as f:
        f.write("lat lon elevation" + '\n')
    write_con(x_surface,y_surface,z_surface)


