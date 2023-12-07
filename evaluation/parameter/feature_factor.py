"""
    This file will evaluate the best value of parameter - percentage of biased features and irrelevant features.
    The values of factors (biased, irrelevant) can be:
    (0.10, 0.10), (0.10,0.15), (0.10,0.20), (0.15,0.15), (0.15, 0.20), (0.20, 0.20)
"""

import matplotlib.pyplot as plt

x_values = ["(0.1, 0.1)", "(0.1, 0.15)", "(0.1, 0.2)", "(0.15, 0.15)", "(0.15, 0.2)", "(0.2, 0.2)"]
number_values = [4480.8, 6069, 9344.4, 7138.8, 10974.8, 20695]
time_values = [253.75094, 275.2426, 339.63712, 328.20112, 401.93538, 681.90138]
ips_values = [17.65825971, 22.04963912, 27.51289376, 21.75129689, 27.30488667, 30.34896336]

fig, axes = plt.subplots(nrows=3, ncols=1, figsize=(7, 9))

# Instance number
axes[0].plot(x_values, number_values, label='Number')
axes[0].set_title('Quantity of generated instances')
axes[0].set_xlabel('percentage')
axes[0].set_ylabel('Number')
axes[0].set_xticks(x_values)
axes[0].legend()

# Time cost
axes[1].plot(x_values, time_values, label='Time')
axes[1].set_title('Time cost of generated instances')
axes[1].set_xlabel('percentage')
axes[1].set_ylabel('Time(s)')
axes[1].set_xticks(x_values)
axes[1].legend()

# Instance per time
axes[2].plot(x_values, ips_values, label='IPS')
axes[2].set_title('Instances Per Second')
axes[2].set_xlabel('percentage')
axes[2].set_ylabel('IPS')
axes[2].set_xticks(x_values)
axes[2].legend()

plt.tight_layout()

plt.show()

