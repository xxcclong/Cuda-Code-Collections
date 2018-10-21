import matplotlib as mpl
import numpy as np
import matplotlib.pyplot as plt
MAXTHREAD = 33
f = open("temp",'r')
s = f.readline()
x_ax = []
for i in range(MAXTHREAD):
    x_ax.append(i+1)
iter_ = 0
da = [[],[],[],[],[],[]]
while(s):
    ll = s.split(', ')
    da[int(iter_/MAXTHREAD)].append(float(ll[1]))
    iter_+=1
    s = f.readline()
# plt.ylim(0, 700)
plt.title("runtime anasis")
# for i in range(6):
#     for k in range(MAXTHREAD):
#         da[i][k] = da[i][k] / max(da[i])
plt.plot(x_ax, da[0],color='green',label='num = 10000')
# plt.plot(x_ax,da[1],color='skyblue',label='num = 20000')
# plt.plot(x_ax,da[2],color='yellow',label='num = 30000')
# plt.plot(x_ax,da[3],color='red',label='num = 50000')
# plt.plot(x_ax,da[4],color='black',label='num = 70000')
# plt.plot(x_ax,da[5],color='pink',label='num = 100000')
plt.legend()
plt.xlabel('offset')
plt.ylabel('GB/s')
plt.save("offset.jpg")

