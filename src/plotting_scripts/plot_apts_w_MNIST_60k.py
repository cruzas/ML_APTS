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
    './results_csv/results_APTS_W_MNIST_60000_2_cleaned.csv',
    './results_csv/results_APTS_W_MNIST_60000_4_cleaned.csv',
    './results_csv/results_APTS_W_MNIST_60000_6_cleaned.csv'
]

avg_losses_per_epoch = {}
avg_accuracies_per_epoch = {}

for file in files:
    df = pd.read_csv(file, converters={'losses': convert_string_to_list, 'accuracies': convert_string_to_list})

    epoch_average_losses = []
    epoch_average_accuracies = []

    losses_transposed = list(zip(*df['losses']))
    for epoch_losses in losses_transposed:
        epoch_average_losses.append(np.mean(epoch_losses))

    accuracies_transposed = list(zip(*df['accuracies']))
    for epoch_accuracies in accuracies_transposed:
        epoch_average_accuracies.append(np.mean(epoch_accuracies))

    N = file.split("_cleaned")[0].split("_")[-1]
    avg_losses_per_epoch[N] = epoch_average_losses
    avg_accuracies_per_epoch[N] = epoch_average_accuracies

file = "./results_csv/adam_mnist_60000_accuracy_metrics.csv"
df = pd.read_csv(file)
avg_accuracies_adam_per_epoch = df['mean']

file = "./results_csv/adam_mnist_60000_loss_metrics.csv"
df = pd.read_csv(file)
avg_losses_adam_per_epoch = df['mean']

plt.figure(figsize=(10, 6))

ax1 = plt.gca()  # Now primary axis for accuracies
ax2 = ax1.twinx()  # Now secondary axis for losses

# Initialize empty lists for custom legend handles and labels
legend_handles = []
legend_labels = []

epochs = np.arange(1, len(avg_losses_adam_per_epoch) + 1)
# Plot accuracies on ax1
line1, = ax1.plot(epochs, avg_accuracies_adam_per_epoch, color='blue')
# Plot losses on ax2
ax2.plot(epochs, avg_losses_adam_per_epoch, color='blue')
# Add the handle and label for the current N to the custom legend lists
legend_handles.append(line1)
legend_labels.append('Adam(lr:0.0025)')

colours = {'2': 'orange', '4': 'green', '6': 'red'}
for N, accuracies in avg_accuracies_per_epoch.items():
    epochs = np.arange(1, len(accuracies) + 1)
    losses = avg_losses_per_epoch[N]

    # Plot accuracies on ax1
    line1, = ax1.plot(epochs, accuracies, color=colours[N])
    
    # Plot losses on ax2
    ax2.plot(epochs, losses, color=colours[N])

    # Add the handle and label for the current N to the custom legend lists
    legend_handles.append(line1)
    legend_labels.append(f'APTS_W(N:{N})')

ax1.set_xlabel('Iterations', fontsize=24)

ax1.set_ylabel('Avg. accuracy (%)', color='k', fontsize=24)
ax2.set_ylabel('Avg. loss', color='k', fontsize=24)

ax1.set_ylim([80, 100])
ax2.set_ylim([1.4, 2.3])

ax1.tick_params('y', colors='k')
ax2.tick_params('y', colors='k')

ax1.set_yscale('linear')
ax2.set_yscale('linear')

# Set axis ticks font size
ax1.tick_params(axis='both', which='major', labelsize=24)
ax2.tick_params(axis='both', which='major', labelsize=24)

# Use the custom handles and labels for the legend
ax1.legend(legend_handles, legend_labels, loc='center right', fontsize=14)

plt.title('Full data set', fontsize=24)
plt.xlim([0, 100])
plt.savefig(f'./plots/APTS_W_MNIST_60000.png', bbox_inches='tight', dpi=300)
plt.show()

