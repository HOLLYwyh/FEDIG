"""
This paper will evaluate the parameter min_len through experiments.
- The value of min_len can be: 250, 500, 750, 1000, 1250, 1500, 1750, 2000
"""

import matplotlib.pyplot as plt

x_values = [250, 500, 750, 1000, 1250, 1500, 1750, 2000]
number_values = []
time_values = []
ips_values = []

fig, axes = plt.subplots(nrows=3, ncols=1, figsize=(7, 9))

# Instance number
axes[0].plot(x_values, number_values, label='Number')
axes[0].set_title('Quantity of generated instances')
axes[0].set_xlabel('min_len')
axes[0].set_ylabel('Number')
axes[0].set_xticks(x_values)
axes[0].legend()

# Time cost
axes[1].plot(x_values, time_values, label='Time')
axes[1].set_title('Time cost of generated instances')
axes[1].set_xlabel('min_len')
axes[1].set_ylabel('Time(s)')
axes[1].set_xticks(x_values)
axes[1].legend()

# Instance per time
axes[2].plot(x_values, ips_values, label='IPS')
axes[2].set_title('Instances Per Second')
axes[2].set_xlabel('min_len')
axes[2].set_ylabel('IPS')
axes[2].set_xticks(x_values)
axes[2].legend()

plt.tight_layout()

plt.show()
