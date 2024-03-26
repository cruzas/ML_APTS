import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os
import ast

# Define whether plotting accuracies or losses
acc_loss = "accuracies"  # Change to "losses" if you want to plot losses instead

# Data set name
dataset="MNIST"

# Array of minibatch sizes
minibatch_sizes = [10000, 60000]

# Array of subdomains
ns = [15, 30, 60]

# Dynamically generate file names and labels based on subdomains
labels = [f"N:{n}" for n in ns]
results = []

# Read and process each file
for minibatch_size in minibatch_sizes:
    filenames = [f"../results_APTS_W_{dataset}_{minibatch_size}_{n}.csv" for n in ns]
    for filename, label in zip(labels):
        try:
            # Construct file path
            path = os.path.abspath(f"./{filename}")
            # Read CSV file
            df = pd.read_csv(path)
            # Preprocess and compute average metrics
            avg_times, avg_accs = [], []
            for i, row in df.iterrows():
                # Safely convert string representations of lists into actual lists
                times_list = ast.literal_eval(row['cum_times'])
                accs_list = ast.literal_eval(row[acc_loss])
                # Convert lists to NumPy arrays for easier computation
                times_array = np.array(times_list, dtype=np.float32)
                accs_array = np.array(accs_list, dtype=np.float32)
                # Append averages
                avg_times.append(np.mean(times_array))
                avg_accs.append(np.mean(accs_array))
            # Add average results to the results list
            results.append((np.mean(avg_times), np.mean(avg_accs), label))
        except Exception as e:
            print(f"Error processing {filename}: {e}")

# Plotting
for avg_time, avg_acc, label in results:
    plt.plot(avg_time, avg_acc, label=label)
plt.xlabel("Avg. Time (s)")
plt.ylabel(f"Avg. {acc_loss.title()} (%)")
plt.legend()
plt.show()
