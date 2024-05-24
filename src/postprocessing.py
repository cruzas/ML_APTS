import numpy as np
import matplotlib.pyplot as plt
import glob

# Function to load and combine results from trial files
def load_and_combine_trials(file_pattern):
    keys = ['epoch_loss', 'epoch_accuracy', 'epoch_times', 'epoch_usage_times',
            'epoch_num_f_evals', 'epoch_num_g_evals', 'epoch_num_sf_evals', 'epoch_num_sg_evals']
    all_trials = {key: [] for key in keys}
    for file in glob.glob(file_pattern):
        trial_data = np.load(file)
        for key in keys:
            all_trials[key].append(trial_data[key])
    return {key: np.stack(all_trials[key]) for key in keys}

# Function to compute mean and variance for specified metrics
def compute_stats(results, metrics):
    stats = {}
    for metric in metrics:
        stats[f'mean_{metric}'] = np.mean(results[metric], axis=0)
        stats[f'var_{metric}'] = np.var(results[metric], axis=0)
    return stats

# Compute cumulative metrics
def compute_cumulative(stats, keys):
    return {f'cum_{key}': np.cumsum(stats[f'mean_{key}']) for key in keys}

# General plotting function with shaded variance
def plot_metrics(x_values, y_values, y_vars, labels, x_label, y_label, title, yscale=None):
    plt.figure(figsize=(12, 8))
    for x, y, var, label in zip(x_values, y_values, y_vars, labels):
        plt.plot(x, y, marker='o', label=label)
        plt.fill_between(x, y - np.sqrt(var), y + np.sqrt(var), alpha=0.2)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    if yscale:
        plt.yscale(yscale)
    plt.legend()
    plt.title(title)
    plt.show()

# Function to plot accuracy and loss on the same plot with dual y-axes and shaded variance
def plot_accuracy_and_loss(x_values, accuracy_values, loss_values, accuracy_vars, loss_vars, labels, x_label, accuracy_label, loss_label, title):
    # Define color and marker options
    colors = ['tab:blue', 'tab:green', 'tab:red', 'tab:orange', 'tab:purple', 'tab:brown', 'tab:pink', 'tab:gray', 'tab:olive', 'tab:cyan']
    markers = ['o', 's', '^', 'd', 'v', 'p', '*', 'h', 'x', '+']

    # Adjust the number of colors and markers based on the number of optimizers
    num_optimizers = len(x_values)
    colors = colors[:num_optimizers]
    markers = markers[:num_optimizers]

    fig, ax1 = plt.subplots(figsize=(12, 8))

    ax1.set_xlabel(x_label)
    ax1.set_ylabel(accuracy_label)
    for x, acc, var_acc, label, color, marker in zip(x_values, accuracy_values, accuracy_vars, labels, colors, markers):
        ax1.plot(x, acc, marker=marker, label=f'{label}', color=color)
        ax1.fill_between(x, acc - np.sqrt(var_acc), acc + np.sqrt(var_acc), color=color, alpha=0.2)
    ax1.tick_params(axis='y')

    ax2 = ax1.twinx()
    ax2.set_ylabel(loss_label)
    for x, loss, var_loss, label, color, marker in zip(x_values, loss_values, loss_vars, labels, colors, markers):
        ax2.plot(x, loss, linestyle='--', marker=marker, color=color)
        ax2.fill_between(x, loss - np.sqrt(var_loss), loss + np.sqrt(var_loss), color=color, alpha=0.2)
    ax2.tick_params(axis='y')

    fig.tight_layout()
    plt.title(title)
    fig.legend(loc='center right', bbox_to_anchor=(0.9, 0.5))
    plt.show()

# Define the number of epochs
epochs = 1  # Replace with the actual number of epochs if different

# Load and combine results for each optimizer
optimizers = ['SGD', 'Adam', 'APTS_4', 'APTS_8']

# Load and combine results for each optimizer
results = {
    'SGD': load_and_combine_trials('SGD_nl6_cifar10_200_0.01_25_2_t*.npz'),
    'Adam': load_and_combine_trials('Adam_nl6_cifar10_200_0.001_25_2_t*.npz'),
    'APTS_4': load_and_combine_trials('APTS_nl6_cifar10_200_0.001_25_4_t*.npz'),
    'APTS_8': load_and_combine_trials('APTS_nl6_cifar10_200_0.001_25_8_t*.npz'),
}

stats = {opt: compute_stats(results[opt], ['epoch_accuracy', 'epoch_loss', 'epoch_times', 'epoch_usage_times',
                                           'epoch_num_f_evals', 'epoch_num_g_evals', 'epoch_num_sf_evals', 'epoch_num_sg_evals'])
         for opt in optimizers}

cumulative_keys = ['epoch_times', 'epoch_usage_times', 'epoch_num_f_evals', 'epoch_num_g_evals', 
                   'epoch_num_sf_evals', 'epoch_num_sg_evals']
cumulative_stats = {opt: compute_cumulative(stats[opt], cumulative_keys) for opt in optimizers}

# Plotting cumulative metrics
epochs = np.arange(0, stats['SGD']['mean_epoch_accuracy'].shape[0])
plot_accuracy_and_loss([epochs] * len(optimizers),
                       [stats[opt]['mean_epoch_accuracy'] for opt in optimizers],
                       [stats[opt]['mean_epoch_loss'] for opt in optimizers],
                       [stats[opt]['var_epoch_accuracy'] for opt in optimizers],
                       [stats[opt]['var_epoch_loss'] for opt in optimizers],
                       optimizers, 'Epochs', 'Mean Accuracy (%)', 'Mean Loss', 'Accuracy and Loss vs Epochs')

# Plot cumulative times for each optimizer
plot_accuracy_and_loss([cumulative_stats[opt]['cum_epoch_times'] for opt in optimizers],
                       [stats[opt]['mean_epoch_accuracy'] for opt in optimizers],
                       [stats[opt]['mean_epoch_loss'] for opt in optimizers],
                       [stats[opt]['var_epoch_accuracy'] for opt in optimizers],
                       [stats[opt]['var_epoch_loss'] for opt in optimizers],
                       optimizers, 'Cumulative Times', 'Mean Accuracy', 'Mean Loss', 'Accuracy and Loss vs Cumulative Times')

# Plot cumulative usage times for each optimizer
plot_accuracy_and_loss([cumulative_stats[opt]['cum_epoch_usage_times'] for opt in optimizers],
                       [stats[opt]['mean_epoch_accuracy'] for opt in optimizers],
                       [stats[opt]['mean_epoch_loss'] for opt in optimizers],
                       [stats[opt]['var_epoch_accuracy'] for opt in optimizers],
                       [stats[opt]['var_epoch_loss'] for opt in optimizers],
                       optimizers, 'Cumulative Usage Times', 'Mean Accuracy', 'Mean Loss', 'Accuracy and Loss vs Cumulative Usage Times')

# Plot cumulative function evaluations for each optimizer
plot_metrics([cumulative_stats[opt]['cum_epoch_num_f_evals'] for opt in optimizers],
             [cumulative_stats[opt]['cum_epoch_num_g_evals'] for opt in optimizers],
             [cumulative_stats[opt]['cum_epoch_num_sf_evals'] for opt in optimizers],
             optimizers, 'Cumulative Function Evaluations', 'Mean Function Evaluations', 'Function Evaluations vs Cumulative Function Evaluations')

# Plot cumulative gradient evaluations for each optimizer
plot_metrics([cumulative_stats[opt]['cum_epoch_num_g_evals'] for opt in optimizers],
             [cumulative_stats[opt]['cum_epoch_num_sg_evals'] for opt in optimizers],
             [cumulative_stats[opt]['cum_epoch_num_sf_evals'] for opt in optimizers],
             optimizers, 'Cumulative Gradient Evaluations', 'Mean Gradient Evaluations', 'Gradient Evaluations vs Cumulative Gradient Evaluations')