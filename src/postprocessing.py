import numpy as np

# Load the data from the npz file
# loaded_results = np.load('SGD_cifar10_200_0.1_50_5.npz')
loaded_results = np.load('Adam_cifar10_200_0.001_50_5.npz')

# Initialize a dictionary to store the results
results = {key: loaded_results[key] for key in loaded_results}

# Initialize a dictionary to store the computed statistics
stats_results = {}

# Iterate over each key in the results dictionary
for key, array in results.items():
    # Compute the required statistics
    average = np.mean(array)
    minimum = np.min(array)
    maximum = np.max(array)
    median = np.median(array)
    std_dev = np.std(array)
    variance = np.var(array)
    
    # Store the computed statistics in the stats_results dictionary
    stats_results[key] = {
        'average': average,
        'min': minimum,
        'max': maximum,
        'median': median,
        'std_dev': std_dev,
        'variance': variance
    }

# Print the results
for key, stats in stats_results.items():
    print(f"Statistics for {key}:")
    for stat_name, value in stats.items():
        print(f"  {stat_name}: {value}")
    print()

