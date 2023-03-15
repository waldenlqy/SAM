import numpy as np	
from matplotlib import pyplot as plt

#plt.rcParams["font.family"] = "Times New Roman"
#simulate data
#xs = range(100)
xs = [50, 100, 200, 500, 1000]
#xs = np.arange(5)
youtube = [0.496, 0.496, 0.496, 0.496, 0.63]
sasrec = [1.425, 2.637,8.4, 14.67,None]
din = [0.62, 0.62, 0.62, 0.754, 1.037]
dien = [0.632, 0.632, 1.438, 8.95, 11.23]
mimn = [0.632, 0.632, 7.68,12.45, None]
ubr = [0.498, 0.498, 0.907, 2.52, 4.659]
imn_3p = [0.63, 0.63, 0.63,0.899, 1.44]


plt.subplot2grid((10,10), (0,0), 9, 10)

#plot data
plt.plot(xs, youtube, lw=2, linestyle='solid', markersize=6, marker='o', color='mediumaquamarine', markerfacecolor='mediumaquamarine',markeredgecolor='mediumaquamarine')
plt.plot(xs, din, lw=2, linestyle='--', markersize=6,marker='s', color='orange', markerfacecolor='orange',markeredgecolor='orange')
plt.plot(xs, dien, lw=2, linestyle='solid', markersize=8,marker='*', color='mediumslateblue', markerfacecolor='mediumslateblue',markeredgecolor='mediumslateblue')
plt.plot(xs, sasrec, lw=2, linestyle='solid',markersize=8, marker='v', color='slategray', markerfacecolor='slategray',markeredgecolor='slategray')
plt.plot(xs, mimn, lw=2, linestyle='--', markersize=6,marker='p', color='black', markerfacecolor='black',markeredgecolor='black')
plt.plot(xs, ubr, lw=2, linestyle='--',markersize=8, marker='*', color='mediumblue', markerfacecolor='mediumblue',markeredgecolor='mediumblue')
#plt.plot(xs, imn_2p, lw=2, linestyle='solid', markersize=6,marker='8', color='orchid', markerfacecolor='orchid',markeredgecolor='orchid')
plt.plot(xs, imn_3p, lw=2, linestyle='solid', markersize=6,marker='D', color='r', markerfacecolor='r',markeredgecolor='r')
plt.axhline(y=7.5, color='orchid', linestyle='dotted', lw=3)
#label axes
plt.xscale('log')
plt.xticks(xs,xs)
plt.xlabel("User Sequence Length", fontsize=19)
plt.ylabel("Peak GPU Memory (GiB)", fontsize=19)
#define offsets
xmin = min(xs)
xmax = max(xs)
x_range = xmax - xmin
x_start = xmin - x_range/25.
x_end = xmax + x_range/25.
x_start = 0
x_end = 1100

ymin = 0
ymax = 15
y_range = ymax - ymin
y_start = ymin - y_range/25.
y_end = ymax + y_range/25.

#define axes with offsets
plt.axis([x_start, x_end, y_start, y_end])

#plot axes (black with line width of 4)
plt.axvline(x=x_start, color="k", lw=3)
plt.axhline(y=y_start, color="k", lw=3)

#plot ticks
plt.tick_params(direction="out", top=False, right=False, length=12, width=2, pad=10, labelsize=16)
plt.legend(["YouTube DNN", "DIN", "DIEN", "SASRec", "MIMN", "UBR4CTR", "SAM 3P", "GPU Memory Limit"],loc='best')


plt.savefig("line")
plt.show()	
