import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from ast import literal_eval

def convert_string_to_list(string):
    try:
        return literal_eval(string)
    except (ValueError, SyntaxError) as e:
        print(f"Failed to convert: {string} (Error: {e})")
        return []

files = [
    './results_csv/results_APTS_W_CIFAR10_10000_2_cleaned.csv',
    './results_csv/results_APTS_W_CIFAR10_10000_4_cleaned.csv',
    './results_csv/results_APTS_W_CIFAR10_10000_6_cleaned.csv'
]

avg_losses_per_epoch = {}
avg_accuracies_per_epoch = {}
avg_cum_times_per_epoch = {}

for file in files:
    df = pd.read_csv(file, converters={'losses': convert_string_to_list, 'accuracies': convert_string_to_list, 'cum_times': convert_string_to_list})

    epoch_average_losses = []
    epoch_average_accuracies = []
    epoch_average_cum_times = []

    losses_transposed = list(zip(*df['losses']))
    for epoch_losses in losses_transposed:
        epoch_average_losses.append(np.mean(epoch_losses))

    accuracies_transposed = list(zip(*df['accuracies']))
    for epoch_accuracies in accuracies_transposed:
        epoch_average_accuracies.append(np.mean(epoch_accuracies))

    cum_times_transposed = list(zip(*df['cum_times']))
    for epoch_cum_times in cum_times_transposed:
        epoch_average_cum_times.append(np.mean(epoch_cum_times))

    N = file.split("_cleaned")[0].split("_")[-1]
    avg_losses_per_epoch[N] = epoch_average_losses
    avg_accuracies_per_epoch[N] = epoch_average_accuracies
    avg_cum_times_per_epoch[N] = epoch_average_cum_times

# Print final time for each N
for N, times in avg_cum_times_per_epoch.items():
    print(f'N: {N}, Final time: {times[-1]}')

plt.figure(figsize=(10, 6))

ax1 = plt.gca()  # Now primary axis for accuracies

# Initialize empty lists for custom legend handles and labels
legend_handles = []
legend_labels = []

colours = {'2': 'orange', '4': 'green', '6': 'red'}
for N, accuracies in avg_accuracies_per_epoch.items():
    times = avg_cum_times_per_epoch[N]

    # Plot accuracies on ax1
    line1, = ax1.plot(times, accuracies, color=colours[N])
    
    # Add the handle and label for the current N to the custom legend lists
    legend_handles.append(line1)
    legend_labels.append(f'APTS_W(N:{N})')

ax1.set_xlabel('Avg. time (s)', fontsize=24)

ax1.set_ylabel('Avg. accuracy (%)', color='k', fontsize=24)

ax1.set_xlim([0, 3000])

ax1.set_ylim([0, 100])

ax1.tick_params('y', colors='k')

ax1.set_yscale('linear')

# Set axis ticks font size
ax1.tick_params(axis='both', which='major', labelsize=24)

# Use the custom handles and labels for the legend
ax1.legend(legend_handles, legend_labels, loc='upper right', fontsize=14)

plt.title('5 minibatches with 1% overlap', fontsize=24)
plt.xlim([0, 3000])
plt.savefig(f'./plots/times_APTS_W_CIFAR10_10000.png', bbox_inches='tight', dpi=300)
plt.show()

