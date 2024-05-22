import numpy as np
import matplotlib.pyplot as plt

# Function to compute average and variance per epoch for both accuracy and loss
def compute_average_and_variance_per_epoch(results):
    mean_accuracy = np.mean(results['epoch_accuracy'], axis=0)
    variance_accuracy = np.var(results['epoch_accuracy'], axis=0)
    mean_loss = np.mean(results['epoch_loss'], axis=0)
    variance_loss = np.var(results['epoch_loss'], axis=0)
    mean_times = np.mean(results['epoch_times'], axis=0)
    mean_usage_times = np.mean(results['epoch_usage_times'], axis=0)
    mean_g_evals = np.mean(results['epoch_num_g_evals'], axis=0)
    variance_g_evals = np.var(results['epoch_num_g_evals'], axis=0)

    # Return a dictionary for easy access
    return {'accuracy': mean_accuracy, 'var_accuracy': variance_accuracy, 'loss': mean_loss, 'var_loss': variance_loss, 'times': mean_times, 'usage_times': mean_usage_times, 'num_g_evals': mean_g_evals, 'var_num_g_evals': variance_g_evals}

# Load the data from the npz files
sgd_results = np.load('SGD_cifar10_200_0.1_50_5.npz')
adam_results = np.load('Adam_cifar10_200_0.001_50_5.npz')
apts_results = np.load('APTS_cifar10_200_0.001_50_5.npz')

# Compute average accuracy and loss per epoch and their variances
sgd_mvs = compute_average_and_variance_per_epoch(sgd_results)
adam_mvs = compute_average_and_variance_per_epoch(adam_results)
apts_mvs = compute_average_and_variance_per_epoch(apts_results)

# Compute cumulative times
sgd_cum_times = np.cumsum(sgd_mvs['times'])
adam_cum_times = np.cumsum(adam_mvs['times'])
apts_cum_times = np.cumsum(apts_mvs['times'])

# Compute cumulative usage times
sgd_cum_usage_times = np.cumsum(sgd_mvs['usage_times'])
adam_cum_usage_times = np.cumsum(adam_mvs['usage_times'])
apts_cum_usage_times = np.cumsum(apts_mvs['usage_times'])

# Compute cumulative number of gradient evaluations
sgd_cum_g_evals = np.cumsum(sgd_mvs['num_g_evals'])
adam_cum_g_evals = np.cumsum(adam_mvs['num_g_evals'])
apts_cum_g_evals = np.cumsum(apts_mvs['num_g_evals'])

# General plotting function
def plot_accuracy_and_loss(x_sgd, x_adam, x_apts, x_label):
    fig, ax1 = plt.figure(figsize=(12, 8)), plt.gca()

    # Define the variables
    sgd_avg_accuracy = sgd_mvs['accuracy']
    sgd_var_accuracy = sgd_mvs['var_accuracy']
    adam_avg_accuracy = adam_mvs['accuracy']
    adam_var_accuracy = adam_mvs['var_accuracy']
    apts_avg_accuracy = apts_mvs['accuracy']
    apts_var_accuracy = apts_mvs['var_accuracy']

    # Plot accuracy on the left y-axis
    ax1.plot(x_sgd, sgd_avg_accuracy, marker='o', color='blue')
    ax1.fill_between(x_sgd, sgd_avg_accuracy - np.sqrt(sgd_var_accuracy), sgd_avg_accuracy + np.sqrt(sgd_var_accuracy), color='blue', alpha=0.2)

    ax1.plot(x_adam, adam_avg_accuracy, marker='s', color='green')
    ax1.fill_between(x_adam, adam_avg_accuracy - np.sqrt(adam_var_accuracy), adam_avg_accuracy + np.sqrt(adam_var_accuracy), color='green', alpha=0.2)
    
    ax1.plot(x_apts, apts_avg_accuracy, marker='^', color='red')
    ax1.fill_between(x_apts, apts_avg_accuracy - np.sqrt(apts_var_accuracy), apts_avg_accuracy + np.sqrt(apts_var_accuracy), color='red', alpha=0.2)

    sgd_avg_loss = sgd_mvs['loss']
    sgd_var_loss = sgd_mvs['var_loss']
    adam_avg_loss = adam_mvs['loss']
    adam_var_loss = adam_mvs['var_loss']
    apts_avg_loss = apts_mvs['loss']
    apts_var_loss = apts_mvs['var_loss']

    # Plot loss on the right y-axis
    ax2 = ax1.twinx()
    ax2.plot(x_sgd, sgd_avg_loss, label='SGD', linestyle='--', marker='o', color='blue')
    ax2.fill_between(x_sgd, sgd_avg_loss - np.sqrt(sgd_var_loss), sgd_avg_loss + np.sqrt(sgd_var_loss), color='blue', alpha=0.1)

    ax2.plot(x_adam, adam_avg_loss, label='Adam', linestyle='--', marker='s', color='green')
    ax2.fill_between(x_adam, adam_avg_loss - np.sqrt(adam_var_loss), adam_avg_loss + np.sqrt(adam_var_loss), color='green', alpha=0.1)

    ax2.plot(x_apts, apts_avg_loss, label='APTS', linestyle='--', marker='^', color='red')
    ax2.fill_between(x_apts, apts_avg_loss - np.sqrt(apts_var_loss), apts_avg_loss + np.sqrt(apts_var_loss), color='red', alpha=0.1)

    ax2.set_ylabel('Average Loss')
    ax2.legend(loc='center right')

    plt.show()

# Plot accuracy and loss versus number of epochs
epochs = np.arange(0, sgd_mvs['accuracy'].shape[0])
# plot_accuracy_and_loss(epochs, epochs, epochs, 'Epochs')

# # Plot accuracy and loss versus cumulative epoch times
# plot_accuracy_and_loss(sgd_cum_times, adam_cum_times, apts_cum_times, 'Cumulative Epoch Times')

# Plot accuracy and loss versus cumulative usage times
# plot_accuracy_and_loss(sgd_cum_usage_times, adam_cum_usage_times, apts_cum_usage_times, 'Cumulative Usage Times')

# Plot number of gradient evaluations versus number of epochs
plot_accuracy_and_loss(sgd_cum_g_evals, adam_cum_g_evals, apts_cum_g_evals, 'Cumulative Number of Gradient Evaluations')