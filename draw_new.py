import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colors
import os

files = os.listdir("./Result")
PSNRs = []
Succeed_rates = []
data = []

for file in files:
    path = "./Result/" + file
    lines = list(open(path, 'r'))
    PSNR = []
    Succeed_rate = []
    for line in lines:
        line = line.split(' ')
        if(float(line[12].split('\n')[0]) > 30.0):
            PSNR.append(float(line[12].split('\n')[0]))
            Succeed_rate.append(int(line[9].split('/')[0]))
    PSNRs.append(PSNR)
    Succeed_rates.append(Succeed_rate)

fig, ax = plt.subplots()
my_colors = list(colors.CSS4_COLORS.keys())
for i in range(0, len(PSNRs)):
    plt.plot(PSNRs[i],Succeed_rates[i],linestyle='-',color=my_colors[i+10], label=files[i])

plt.xlabel('PSNR')
plt.ylabel('ASR')
ax.invert_xaxis() 
plt.legend()
plt.show()

