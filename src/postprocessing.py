import numpy as np
import matplotlib.pyplot as plt
import glob

# Function to compute average and variance per epoch for both accuracy and loss
def compute_average_and_variance_per_epoch(results):
    # Accuracy
    mean_accuracy = np.mean(results['epoch_accuracy'], axis=0)
    variance_accuracy = np.var(results['epoch_accuracy'], axis=0)
    # Loss
    mean_loss = np.mean(results['epoch_loss'], axis=0)
    variance_loss = np.var(results['epoch_loss'], axis=0)
    # Times per epoch
    mean_times = np.mean(results['epoch_times'], axis=0)
    mean_usage_times = np.mean(results['epoch_usage_times'], axis=0)
    # Number of function and gradient evaluations per epoch
    mean_f_evals = np.mean(results['epoch_num_f_evals'], axis=0)
    variance_f_evals = np.var(results['epoch_num_f_evals'], axis=0)
    mean_g_evals = np.mean(results['epoch_num_g_evals'], axis=0)
    variance_g_evals = np.var(results['epoch_num_g_evals'], axis=0)
    # Number of subdomain function and gradient evaluations per epoch
    mean_sf_evals = np.mean(results['epoch_num_sf_evals'], axis=0)
    variance_sf_evals = np.var(results['epoch_num_sf_evals'], axis=0)
    mean_sg_evals = np.mean(results['epoch_num_sg_evals'], axis=0)
    variance_sg_evals = np.var(results['epoch_num_sg_evals'], axis=0)

    # Return a dictionary for easy access
    return {
        'accuracy': mean_accuracy, 'var_accuracy': variance_accuracy,
        'loss': mean_loss, 'var_loss': variance_loss,
        'times': mean_times, 'usage_times': mean_usage_times,
        'num_f_evals': mean_f_evals, 'var_num_f_evals': variance_f_evals,
        'num_g_evals': mean_g_evals, 'var_num_g_evals': variance_g_evals,
        'num_sf_evals': mean_sf_evals, 'var_num_sf_evals': variance_sf_evals,
        'num_sg_evals': mean_sg_evals, 'var_num_sg_evals': variance_sg_evals
    }

# Function to load and combine results from trial files
def load_and_combine_trials(file_pattern, epochs):
    """
    Loads and combines results from trial files matching the given pattern.

    Args:
    - file_pattern (str): Pattern to match the trial files (e.g., 'SGD_nl6_cifar10_200_0.01_1_2_t*.npz').
    - epochs (int): Number of epochs.

    Returns:
    - dict: A dictionary containing combined results with keys corresponding to different metrics.
            Each value is a NumPy array of shape (trials, epochs).
    """
    # Initialize empty lists to accumulate data from each trial
    all_trials = {key: [] for key in [
        'epoch_loss', 'epoch_accuracy', 'epoch_times', 'epoch_usage_times',
        'epoch_num_f_evals', 'epoch_num_g_evals', 'epoch_num_sf_evals', 'epoch_num_sg_evals'
    ]}

    # Loop over all files matching the pattern
    for file in glob.glob(file_pattern):
        # Load the trial data
        trial_data = np.load(file)
        
        # Append the data to the corresponding lists in all_trials
        for key in all_trials.keys():
            all_trials[key].append(trial_data[key])

    # Convert lists to NumPy arrays with the appropriate shape (trials, epochs)
    for key in all_trials.keys():
        all_trials[key] = np.stack(all_trials[key])

    return all_trials

# Define the number of epochs
epochs = 1  # Replace with the actual number of epochs if different

# Load and combine results for each optimizer
sgd_results = load_and_combine_trials('SGD_nl6_cifar10_200_0.01_1_2_t*.npz', epochs)
adam_results = load_and_combine_trials('Adam_nl6_cifar10_200_0.001_1_2_t*.npz', epochs)
apts_results = load_and_combine_trials('APTS_nl6_cifar10_200_0.001_1_2_t*.npz', epochs)

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

# Compute cumulative number of function and gradient evaluations
sgd_cum_f_evals = np.cumsum(sgd_mvs['num_f_evals'])
adam_cum_f_evals = np.cumsum(adam_mvs['num_f_evals'])
apts_cum_f_evals = np.cumsum(apts_mvs['num_f_evals'])
sgd_cum_g_evals = np.cumsum(sgd_mvs['num_g_evals'])
adam_cum_g_evals = np.cumsum(adam_mvs['num_g_evals'])
apts_cum_g_evals = np.cumsum(apts_mvs['num_g_evals'])

# Compute cumulative number of subdomain function and gradient evaluations
sgd_cum_sf_evals = np.cumsum(sgd_mvs['num_sf_evals'])
adam_cum_sf_evals = np.cumsum(adam_mvs['num_sf_evals'])
apts_cum_sf_evals = np.cumsum(apts_mvs['num_sf_evals'])
sgd_cum_sg_evals = np.cumsum(sgd_mvs['num_sg_evals'])
adam_cum_sg_evals = np.cumsum(adam_mvs['num_sg_evals'])
apts_cum_sg_evals = np.cumsum(apts_mvs['num_sg_evals'])

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
    ax1.set_xlabel(x_label)
    ax1.set_ylabel('Mean Accuracy')

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

    ax2.set_ylabel('Mean Loss')
    ax2.legend(loc='center right')

    plt.title('Accuracy and Loss versus ' + x_label)
    plt.show()

# Plot accuracy and loss versus number of epochs
epochs = np.arange(0, sgd_mvs['accuracy'].shape[0])
# plot_accuracy_and_loss(epochs, epochs, epochs, 'Epochs')

# # Plot accuracy and loss versus cumulative epoch times
# plot_accuracy_and_loss(sgd_cum_times, adam_cum_times, apts_cum_times, 'Cumulative Epoch Times')

# Plot accuracy and loss versus cumulative usage times
# plot_accuracy_and_loss(sgd_cum_usage_times, adam_cum_usage_times, apts_cum_usage_times, 'Cumulative Usage Times')

# # Plot number of function evaluations versus number of epochs
# plot_accuracy_and_loss(sgd_cum_f_evals, adam_cum_f_evals, apts_cum_f_evals, 'Cumulative Number of Function Evaluations')

# Plot number of gradient evaluations versus number of epochs
# plot_accuracy_and_loss(sgd_cum_g_evals, adam_cum_g_evals, apts_cum_g_evals, 'Cumulative Number of Gradient Evaluations')

# Plot number of subdomain function evaluations versus number of epochs
# plot_accuracy_and_loss(sgd_cum_sf_evals, adam_cum_sf_evals, apts_cum_sf_evals, 'Cumulative Number of Subdomain Function Evaluations')

# Plot number of subdomain gradient evaluations versus number of epochs
# plot_accuracy_and_loss(sgd_cum_sg_evals, adam_cum_sg_evals, apts_cum_sg_evals, 'Cumulative Number of Subdomain Gradient Evaluations')

# Plot the correlation between num_sf_evals and num_f_evals to see how they are related over epochs and trials.
# plt.figure(figsize=(12, 8))
# plt.scatter(apts_cum_f_evals, apts_cum_sf_evals, marker='^', color='red', label='APTS')
# plt.xlabel('Cumulative Number of Function Evaluations')
# plt.ylabel('Cumulative Number of Subdomain Function Evaluations')
# plt.legend()
# plt.title('Cumulative Number of Subdomain Function Evaluations versus Cumulative Number of Function Evaluations')
# plt.show()

# Plot the correlation between num_sg_evals and num_g_evals to see how they are related over epochs and trials.
# plt.figure(figsize=(12, 8))
# plt.scatter(apts_cum_g_evals, apts_cum_sg_evals, marker='^', color='red', label='APTS')
# plt.xlabel('Cumulative Number of Gradient Evaluations')
# plt.ylabel('Cumulative Number of Subdomain Gradient Evaluations')
# plt.legend()
# plt.title('Cumulative Number of Subdomain Gradient Evaluations versus Cumulative Number of Gradient Evaluations')
# plt.show()

# Plot the number of function evaluations versus number of epochs
# plt.figure(figsize=(12, 8))
# plt.plot(epochs, sgd_cum_f_evals, marker='o', color='blue', label='SGD')
# plt.plot(epochs, adam_cum_f_evals, marker='s', color='green', label='Adam')
# plt.plot(epochs, apts_cum_f_evals, marker='^', color='red', label='APTS')
# plt.xlabel('Number of Epochs')
# plt.ylabel('Cumulative Number of Function Evaluations')
# plt.legend()
# plt.title('Cumulative Number of Function Evaluations versus Number of Epochs')
# plt.show()

# Plot the number of gradient evaluations versus number of epochs
# plt.figure(figsize=(12, 8))
# plt.plot(epochs, sgd_cum_g_evals, marker='o', color='blue', label='SGD')
# plt.plot(epochs, adam_cum_g_evals, marker='s', color='green', label='Adam')
# plt.plot(epochs, apts_cum_g_evals, marker='^', color='red', label='APTS')
# plt.xlabel('Number of Epochs')
# plt.ylabel('Cumulative Number of Gradient Evaluations')
# plt.legend()
# plt.title('Cumulative Number of Gradient Evaluations versus Number of Epochs')
# plt.show()

# Plot the ratio of the number of function evaluations of APTS to Adam versus number of epochs
# plt.figure(figsize=(12, 8))
# plt.plot(epochs, apts_cum_f_evals / adam_cum_f_evals, marker='^', color='red', label='APTS/Adam')
# plt.xlabel('Number of Epochs')
# plt.ylabel('Ratio of Cumulative Number of Function Evaluations')
# plt.yscale('log')  # Setting the y-axis to logarithmic scale
# plt.legend()
# plt.title('Ratio of Cumulative Number of Function Evaluations of APTS to Adam versus Number of Epochs')
# plt.show()

# Plot the ratio of the number of gradient evaluations of APTS to Adam versus number of epochs
# plt.figure(figsize=(12, 8))
# plt.plot(epochs, apts_cum_g_evals / adam_cum_g_evals, marker='^', color='red', label='APTS/Adam')
# plt.xlabel('Number of Epochs')
# plt.ylabel('Ratio of Cumulative Number of Gradient Evaluations')
# plt.legend()
# plt.title('Ratio of Cumulative Number of Gradient Evaluations of APTS to Adam versus Number of Epochs')
# plt.show()

# Plot the ratio of the number of function evaluations of APTS to number of subdomain function evaluations of APTS versus number of epochs
# plt.figure(figsize=(12, 8))
# plt.plot(epochs, apts_cum_f_evals / apts_cum_sf_evals, marker='^', color='red', label='APTS')
# plt.xlabel('Number of Epochs')
# plt.ylabel('Ratio of Cumulative Number of Function Evaluations')
# plt.legend()
# plt.title('Ratio of Cumulative Number of Function Evaluations of APTS to Number of Subdomain Function Evaluations of APTS versus Number of Epochs')
# plt.show()

# Plot the ratio of the number of gradient evaluations of APTS to number of subdomain gradient evaluations of APTS versus number of epochs
# plt.figure(figsize=(12, 8))
# plt.plot(epochs, apts_cum_g_evals / apts_cum_sg_evals, marker='^', color='red', label='APTS')
# plt.xlabel('Number of Epochs')
# plt.ylabel('Ratio of Cumulative Number of Gradient Evaluations')
# plt.legend()
# plt.title('Ratio of Cumulative Number of Gradient Evaluations of APTS to Number of Subdomain Gradient Evaluations of APTS versus Number of Epochs')
# plt.show()

