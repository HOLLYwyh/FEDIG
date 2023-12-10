"""
This paper will evaluate the parameter min_len through experiments.
- The value of min_len can be: 250, 500, 750, 1000, 1250, 1500, 1750, 2000
"""

import matplotlib.pyplot as plt

x_values = [250, 500, 750, 1000, 1250, 1500, 1750, 2000]
number_values = [9289.4, 9305.8, 9308.4, 9310.4, 9327.4, 9317.6, 9340.8, 9304.8]
time_values = [307.83954, 317.3964, 329.23108, 337.38886, 339.31932, 337.54764, 337.49174, 338.1877]
ips_values = [30.17611058, 29.31917312, 28.27315088, 27.59545766, 27.48856151, 27.60380727, 27.67712182, 27.51371502]

fig, axes = plt.subplots(nrows=3, ncols=1, figsize=(7, 9))

# Instance number
axes[0].plot(x_values, number_values, label='Number', marker='D', color='darkmagenta')
axes[0].set_title('Quantity of generated instances')
axes[0].set_xlabel('min_len')
axes[0].set_ylabel('Number')
axes[0].set_xticks(x_values)
axes[0].legend()

# Time cost
axes[1].plot(x_values, time_values, label='Time', marker='D', color='darkmagenta')
axes[1].set_title('Time cost of generated instances')
axes[1].set_xlabel('min_len')
axes[1].set_ylabel('Time(s)')
axes[1].set_xticks(x_values)
axes[1].legend()

# Instance per time
axes[2].plot(x_values, ips_values, label='IPS', marker='D', color='darkmagenta')
axes[2].set_title('Instances Per Second')
axes[2].set_xlabel('min_len')
axes[2].set_ylabel('IPS')
axes[2].set_xticks(x_values)
axes[2].legend()

plt.tight_layout()

plt.show()
