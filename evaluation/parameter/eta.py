"""
    This file will evaluate the best value of parameter eta η.
     - The value of η can be: 0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6
     - 0.7, 0.8, 0.9, 1.0
"""

import matplotlib.pyplot as plt

x_values = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
number_values = [9820, 9477.8, 9343.6, 8850.2, 8513.4, 7361.6, 6799.8, 6247.8, 4621.2, 2849.8, 2071.2]
time_values = [401.24426, 353.36988, 345.41398, 341.98108, 360.4856,
               345.92308, 360.6842, 367.74118, 353.44962, 325.83972, 283.27436]
ips_values = [24.47387036, 26.82118804, 27.05043959, 25.87920946, 23.61647733,
              21.28103161, 18.8525031, 16.98966648, 13.07456491, 8.746017827, 7.3116395]

plt.figure(figsize=(9, 7))

plt.subplot(2, 2, 1)
plt.plot(x_values, number_values, label='Number', marker='o')
plt.title('Quantity')
plt.xlabel('η value')
plt.ylabel('Number')
plt.xticks(x_values)
plt.legend()

plt.subplot(2, 2, 2)
plt.plot(x_values, time_values, label='Time', marker='o')
plt.title('Time cost')
plt.xlabel('η value')
plt.ylabel('Time(s)')
plt.xticks(x_values)
plt.legend()

plt.subplot(2, 2, 3)
plt.plot(x_values, ips_values, label='IPS', marker='o')
plt.title('Instances Per Second')
plt.xlabel('η value')
plt.ylabel('IPS')
plt.xticks(x_values)
plt.legend()


# 调整布局
plt.tight_layout()

# 显示图形
plt.show()

