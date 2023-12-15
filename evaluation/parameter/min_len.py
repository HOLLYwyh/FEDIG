"""
This paper will evaluate the parameter min_len through experiments.
- The value of min_len can be: 250, 500, 750, 1000, 1250, 1500, 1750, 2000
"""

import matplotlib.pyplot as plt

x_values = [250, 500, 750, 1000, 1250, 1500, 1750, 2000]
number_values = [9289.4, 9305.8, 9308.4, 9310.4, 9327.4, 9317.6, 9340.8, 9304.8]
time_values = [307.83954, 317.3964, 329.23108, 337.38886, 339.31932, 337.54764, 337.49174, 338.1877]
ips_values = [30.17611058, 29.31917312, 28.27315088, 27.59545766, 27.48856151, 27.60380727, 27.67712182, 27.51371502]

plt.figure(figsize=(9, 7))

plt.subplot(2, 2, 1)
plt.plot(x_values, number_values, label='Number', marker='D', color='darkmagenta')
plt.title('Quantity')
plt.xlabel('min_len')
plt.ylabel('Number')
plt.xticks(x_values)
plt.legend()

plt.subplot(2, 2, 2)
plt.plot(x_values, time_values, label='Time', marker='D', color='darkmagenta')
plt.title('Time cost')
plt.xlabel('min_len')
plt.ylabel('Time(s)')
plt.xticks(x_values)
plt.legend()

plt.subplot(2, 2, 3)
plt.plot(x_values, ips_values, label='IPS', marker='D', color='darkmagenta')
plt.title('Instances Per Second')
plt.xlabel('min_len')
plt.ylabel('IPS')
plt.xticks(x_values)
plt.legend()

# 调整布局
plt.tight_layout()

# 显示图形
plt.show()
