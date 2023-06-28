import h5py
from math import radians, cos, sin, asin, sqrt
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import os.path
import numpy as np
import bottom.setting_data.setting as st
import bottom.get_bottom_surface as gbs


# # filepath="E:\\Data\\IceSat-2\\nanhai_atl03\\day\\ATL03_20200419053723_03620701_005_01.h5"
# filepath=r"C:\Users\17826\Desktop\Program\bathymetry\download\ATL03_20181209231549_11050101_005_01\ATL03_20181209231549_11050101_005_01.h5"
# strength="r"
# flight_path="2"
# range_low=21.273
# range_high=21.295
upper, median, lower=gbs.get_gbs()

filepath,flight_path,strength,range_low,range_high,interval_x,interval_z=st.getsetting()
filename=os.path.split(filepath)[1]
beam=str(flight_path)+strength
# strength="l"
# flight_path="1"
# range_low=16.169
# range_high=16.182

x,y,z=[],[],[]
sub_x,sub_y,sub_z=[],[],[]
op_x,op_y,op_z=[],[],[]
with h5py.File(filepath, mode='r') as f:
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
    quality = f['/gt'+str(flight_path)+strength+'/heights/quality_ph']
    quality0 = quality[:]
    confidence = f['/gt'+str(flight_path)+strength+'/heights/signal_conf_ph']
    confidence4 = confidence[:]
    for i in range(len(temp)):
        if(latitude[i]>range_low and latitude[i] < range_high):
            x.append(latitude[i])
            y.append(longitude[i])
            z.append(temp[i])
    for i in range(len(x)):
        if(z[i]<lower and z[i]>median-50 ):
            sub_x.append(x[i])
            sub_y.append(y[i])
            sub_z.append(z[i])

def func(x, a, b, c):
    return a * np.exp(-b*x) + c

shuixia_1=[]
for i in range(len(sub_z)-1):
    a=abs(sub_z[i+1]-sub_z[i])
    shuixia_1.append(a)
n_shuixia1=plt.hist(shuixia_1,bins=100,rwidth=0.9, density=True)
pro=n_shuixia1[0]*np.diff(n_shuixia1[1])
h_cha=np.array(n_shuixia1[1])
h_cha=h_cha[:pro.shape[0]]
# 声明待拟合的数据
xdata = h_cha
ydata = pro
# 拟合数据并展示曲线
popt, pcov = curve_fit(func, xdata, ydata)
y2=func(xdata, *popt)-popt[2]
for i in range(len(y2)-1):
    if abs(y2[i]-0)>1e-2:
#         print(xdata[i])
        bb=xdata[i]
print("第一次去噪y2[i]-0:",y2[i]-0)
yuzhi1=bb

shuixia_quzao_z=[]
shuixia_quzao_x=[]
shuixia_quzao_y=[]
for i in range(len(sub_z)-1):
    if abs(sub_z[i+1]-sub_z[i])<yuzhi1:
        shuixia_quzao_x.append(sub_x[i])
        shuixia_quzao_y.append(sub_y[i])
        shuixia_quzao_z.append(sub_z[i])
plt.figure(figsize=(24, 10))
plt.scatter(x,z,20,'gray')
plt.scatter(shuixia_quzao_x,shuixia_quzao_z,20,'r')
plt.xlabel('latitude (deg)',fontsize=32,family='Times New Roman')
plt.ylabel('height (m)',fontsize=32,family='Times New Roman')
plt.xticks(fontsize=32,family='Times New Roman')
plt.yticks(fontsize=32,family='Times New Roman')
plt.legend(['noise','First denosing'],fontsize=32,markerscale=2,edgecolor='black')
plt.show()



shuixia_2=[]
for i in range(len(sub_z)-1):
    a=abs(sub_z[i+1]-sub_z[i])
    shuixia_2.append(a)
n_shuixia2=plt.hist(shuixia_2,bins=100,rwidth=0.9, density=True)
pro=n_shuixia2[0]*np.diff(n_shuixia2[1])
h_cha=np.array(n_shuixia2[1])
h_cha=h_cha[:pro.shape[0]]
# 声明待拟合的数据
xdata = h_cha
ydata = pro
# 拟合数据并展示曲线
popt, pcov = curve_fit(func, xdata, ydata)
y2=func(xdata, *popt)-popt[2]
for i in range(len(y2)-1):
    if abs(y2[i]-0)>1e-3:
#         print(xdata[i])
        bb=xdata[i]
yuzhi2=bb
print("第二次去噪y2[i]-0:",y2[i]-0)
shuixia_quzao2_x,shuixia_quzao2_y,shuixia_quzao2_z=[],[],[]
for i in range(len(shuixia_quzao_z)-1):
    if abs(shuixia_quzao_z[i+1]-shuixia_quzao_z[i])<yuzhi2:
        shuixia_quzao2_x.append(shuixia_quzao_x[i])
        shuixia_quzao2_y.append(shuixia_quzao_y[i])
        shuixia_quzao2_z.append(shuixia_quzao_z[i])
plt.figure(figsize=(24, 10))
plt.scatter(x,z,20,'gray')
plt.scatter(shuixia_quzao2_x,shuixia_quzao2_z,20,'r')
plt.xlabel('latitude (deg)',fontsize=32,family='Times New Roman')
plt.ylabel('height (m)',fontsize=32,family='Times New Roman')
plt.xticks(fontsize=32,family='Times New Roman')
plt.yticks(fontsize=32,family='Times New Roman')
plt.legend(['noise','Second denosing'],fontsize=32,markerscale=2,edgecolor='black')
plt.show()


shuixia_3=[]
for i in range(len(shuixia_quzao2_z)-1):
    a=abs(shuixia_quzao2_z[i+1]-shuixia_quzao2_z[i])
    shuixia_3.append(a)
plt.figure(figsize=(20, 10))
n_shuixia3=plt.hist(shuixia_3,bins=100,rwidth=0.9, density=True)
plt.cla()
pro=n_shuixia3[0]*np.diff(n_shuixia3[1])
h_cha=np.array(n_shuixia3[1])
h_cha=h_cha[:pro.shape[0]]
# 声明待拟合的数据
xdata = h_cha
ydata = pro
# 拟合数据并展示曲线
popt, pcov = curve_fit(func, xdata, ydata)
y2=func(xdata, *popt)-popt[2]
for i in range(len(y2)-1):
    if abs(y2[i]-0)>1e-3:
#         print(xdata[i])
        bb=xdata[i]
yuzhi3=bb
print("第三次去噪y2[i]-0:",y2[i]-0)
shuixia_quzao3_x,shuixia_quzao3_y,shuixia_quzao3_z=[],[],[]
for i in range(len(shuixia_quzao2_z)-1):
    if abs(shuixia_quzao2_z[i+1]-shuixia_quzao2_z[i])<yuzhi3:
        shuixia_quzao3_x.append(shuixia_quzao2_x[i])
        shuixia_quzao3_y.append(shuixia_quzao2_y[i])
        shuixia_quzao3_z.append(shuixia_quzao2_z[i])
plt.figure(figsize=(24, 10))
plt.scatter(x,z,20,'gray')
plt.scatter(shuixia_quzao3_x,shuixia_quzao3_z,20,'r')
plt.xlabel('latitude (deg)',fontsize=32,family='Times New Roman')
plt.ylabel('height (m)',fontsize=32,family='Times New Roman')
plt.xticks(fontsize=32,family='Times New Roman')
plt.yticks(fontsize=32,family='Times New Roman')
plt.legend(['noise','Third denosing'],fontsize=32,markerscale=2,edgecolor='black')
plt.xticks(fontsize=32,family='Times New Roman')
plt.yticks(fontsize=32,family='Times New Roman')
plt.show()


# shuixia_4=[]
# for i in range(len(shuixia_quzao3_z)-1):
#     a=abs(shuixia_quzao3_z[i+1]-shuixia_quzao3_z[i])
#     shuixia_4.append(a)
# plt.figure(figsize=(20, 10))
# n_shuixia4=plt.hist(shuixia_4,bins=100,rwidth=0.9, density=True)
# plt.cla()
# pro=n_shuixia4[0]*np.diff(n_shuixia4[1])
# h_cha=np.array(n_shuixia4[1])
# h_cha=h_cha[:pro.shape[0]]
# # 声明待拟合的数据
# xdata = h_cha
# ydata = pro
# # 拟合数据并展示曲线
# popt, pcov = curve_fit(func, xdata, ydata)
# y2=func(xdata, *popt)-popt[2]
# for i in range(len(y2)-1):
#     if abs(y2[i]-0)>1e-3:
# #         print(xdata[i])
#         bb=xdata[i]
# yuzhi4=bb
# print("第四次去噪y2[i]-0:",y2[i]-0)
# shuixia_quzao4_x,shuixia_quzao4_y,shuixia_quzao4_z=[],[],[]
# for i in range(len(shuixia_quzao3_z)-1):
#     if abs(shuixia_quzao3_z[i+1]-shuixia_quzao3_z[i])<yuzhi4:
#         shuixia_quzao4_x.append(shuixia_quzao3_x[i])
#         shuixia_quzao4_y.append(shuixia_quzao3_y[i])
#         shuixia_quzao4_z.append(shuixia_quzao3_z[i])
# plt.figure(figsize=(24, 10))
# plt.scatter(x,z,20,'gray')
# plt.scatter(shuixia_quzao4_x,shuixia_quzao4_z,20,'r')
# plt.xlabel('latitude (deg)',fontsize=32,family='Times New Roman')
# plt.ylabel('height (m)',fontsize=32,family='Times New Roman')
# plt.xticks(fontsize=32,family='Times New Roman')
# plt.yticks(fontsize=32,family='Times New Roman')
# plt.legend(['noise','Forth denosing'],fontsize=32,markerscale=2,edgecolor='black')
# plt.xticks(fontsize=32,family='Times New Roman')
# plt.yticks(fontsize=32,family='Times New Roman')
# plt.show()

bottom_txt_name=filename+"_"+beam+"_"+str(round(range_low,3))+"_"+str(round(range_high,3))
filename_ = filename.split(".")[0]
result_path = r'C:\\Users\\17826\\Desktop\Program\\bathymetry\\bottom\\result\\result_demo_12\\AEDTA\\'+ bottom_txt_name + ".txt"
# 头文件字符串
def write_con(x,y,z):
    with open(result_path , "a", encoding='utf-8') as f:
        for i in range(len(x)):
            f.write(str(x[i])+" "+str(y[i])+" "+str(z[i]) + '\n')

# # 生成的数据写入到文件
# with open(result_path, "a", encoding='utf-8') as f:
#     f.write("lat lon elevation"+ '\n')
# write_con(shuixia_quzao3_x,shuixia_quzao3_y,shuixia_quzao3_z)
