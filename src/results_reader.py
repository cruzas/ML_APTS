# in this file we will read the results from the CSV file and plot them
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os, ast
import argparse
import sys
import torch

acc_loss = "accuracies" # or "losses"

path0 = os.path.abspath("./results_APTS_W_2.csv")
path1 = os.path.abspath("./results_APTS_W_15.csv")
path2 = os.path.abspath("./results_APTS_W_30.csv")
path3 = os.path.abspath("./results_APTS_W_60.csv")

paths = [path0, path1, path2, path3]
for path in paths:
    df = pd.read_csv(path)
    df2 = pd.DataFrame()
    temp = [0]*len(df["cum_times"]);temp2 = [0]*len(df["cum_times"])
    df[acc_loss].apply(ast.literal_eval)
    for i in range(len(df["cum_times"])):
        t = df["cum_times"][i][1:-1].strip()
        for j in range(20,1,-1):
            t = t.replace(' '*j,' ')
        temp[i] = np.array(t.split(" "), dtype=np.float32)
        temp2[i] = np.array(df[acc_loss][i][1:-1].replace(' ','').split(","), dtype=np.float32)
    # stack temp into a numpy array and take the vertical mean
    avg_times = np.mean(np.stack(temp), axis=0)
    avg_accs = np.mean(np.stack(temp2), axis=0)
    plt.plot(avg_times, avg_accs, label=path.split("/")[-1].split(".")[0])
    # Add x and y labels
    

plt.show()
    
    


        