import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from IPython.display import clear_output
import pandas as pd

from qiskit_machine_learning.algorithms.classifiers import NeuralNetworkClassifier
from sklearn.model_selection import KFold
from qiskit_algorithms.optimizers import COBYLA, NELDER_MEAD, SLSQP, SPSA

from typing import List


def plot_loss(loss, label='loss', ylim=[0, 1.4], figsize=(12, 6), dpi=300):
    plt.rcParams["figure.figsize"] = figsize
    plt.rcParams['figure.dpi'] = dpi
    plt.plot(loss, 'tab:blue', label=label)
    plt.ylim(ylim[0], ylim[1])
    plt.ylabel('loss')
    plt.xlabel('iterations')
    plt.legend(loc='best')
    plt.show()


def score(predicted, test_labels):
    # arr = []
    predicted = predicted.reshape(-1)
    return np.mean(predicted == test_labels)


# parity maps bitstrings to 0 or 1
def parity(x):
    return "{:b}".format(x).count("1")

# Classification callback


class classification_callback:
    name = "class_callback"

    # Initialise callback for objfun silent collection
    def __init__(self, log_interval=50):
        self.objfun_vals = []
        self.weight_vals = []
        self.log_interval = log_interval
        print('Callback initialted')

    # Find the first minimum objective fun value
    def min_obj(self):
        if self.objfun_vals == []:
            return (-1, 0)
        else:
            minval = min(self.objfun_vals)
            minvals = [(i, v)
                       for i, v in enumerate(self.objfun_vals) if v == minval]
            return minvals[0]

    # Plots the objfun chart
    def plot(self):
        clear_output(wait=True)
        plt.rcParams["figure.figsize"] = (12, 6)
        plt.title("Objective function")
        plt.xlabel("Iteration")
        plt.ylabel("Objective function value")
        plt.plot(range(len(self.objfun_vals)), self.objfun_vals)
        plt.show()

    # Store objective function values and weights and plot
    def graph(self, weights, obj_func_eval):
        self.objfun_vals.append(obj_func_eval)
        self.plot()
        print(f'Step: {len(self.objfun_vals)}')
        print(f'Current value: {obj_func_eval} ')

    # Collects objfun values and prints their values at log intervals
    # When finished the "plot" function can produce the chart
    def collect(self, weights, obj_func_eval):
        self.objfun_vals.append(obj_func_eval)
        self.weight_vals.append(weights)
        current_batch_idx = len(self.objfun_vals)
        if current_batch_idx % self.log_interval == 0:
            prev_batch_idx = current_batch_idx-self.log_interval
            last_batch_min = np.min(
                self.objfun_vals[prev_batch_idx:current_batch_idx])
            print('Prev=', prev_batch_idx, ', Curr=', current_batch_idx)
            print('Classification callback(',
                  current_batch_idx, ') = ', obj_func_eval)

# Exponential Moving Target used to smooth the lines 
def smooth(scalars: List[float], weight: float) -> List[float]:  # Weight between 0 and 1
    last = scalars[0]  # First value in the plot (first timestep)
    smoothed = list()
    for point in scalars:
        smoothed_val = last * weight + (1 - weight) * point  # Calculate smoothed value
        smoothed.append(smoothed_val)                        # Save it
        last = smoothed_val                                  # Anchor the last smoothed value
        
    return smoothed


def plot_many_series(series, smooth_weight=0, d = None, xlabel='Iterations', ylabel='Score', figsize=(12, 6), dpi=300, 
                     title='Series Score vs Iteration', label='Series', xlim=None, ylim=None, pref='d='):
    
    color = ['tab:blue', 'tab:orange' ,'tab:green', 'tab:red', 'tab:purple', 'tab:brown', 'tab:pink', 'tab:gray', 'tab:olive', 'tab:cyan']

    plt.rcParams['figure.figsize'] = figsize
    plt.rcParams['figure.dpi'] = dpi
    
    for c in range(len(series)):
        select = series[c]
        max = smooth(select.max(), smooth_weight)
        min = smooth(select.min(), smooth_weight)
        mean = smooth(select.mean(), smooth_weight)
        
        if d:
            plt.plot(range(0, series[c].shape[1]), mean, color = color[c], label=f'{label} ({c}), {pref}{d[c]}')
        else:
            plt.plot(range(0, series[c].shape[1]), mean, color = color[c], label=f'{label} ({c})')

        plt.fill_between(range(0, series[c].shape[1]), max, min, color = color[c], alpha = 0.2)

    if xlim is not None: plt.xlim(xlim)
    if ylim is not None: plt.ylim(ylim)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.legend(loc='best')

def plot_objfn_range(objective_fn, smooth_weight=0, d = None, xlabel='Iterations', ylabel='Loss', title='Loss Function vs Iteration', label='Method'):
    plot_many_series(objective_fn, smooth_weight=smooth_weight, d = None, xlabel=xlabel, ylabel=ylabel, title=title, label=label)


def plot_score_range(scores, smooth_weight=0, xlabel='Iterations', ylabel='Score', title='Scores vs Iteration'):
    color = {
        'm1': 'tab:blue', 
        'm2': 'tab:orange' ,
        'm3': 'tab:green', 
        'm4':'tab:red'
        }
    for c in scores:
        select = pd.DataFrame(np.reshape(scores[c], (len(scores[c]), scores[c][0].shape[1])))
        
        max = smooth(select.max(), smooth_weight)
        min = smooth(select.min(), smooth_weight)
        mean = smooth(select.mean(), smooth_weight)
        

        plt.plot(range(0, select.shape[1]), mean, color = color[c], label=f'Method {c} Average')

        plt.fill_between(range(0, select.shape[1]), max, min, color = color[c], alpha = 0.2)
    
    plt.ylim(0.2, 1.1)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.legend(loc='lower right')

def result_to_objfun_dataframes(callback_results):
    dataframes = []
    for c in range(len(callback_results)):
        objfn_val_df = pd.DataFrame([callback_results[c][i].objfun_vals for i in range(len(callback_results[c]))])
        # objfn_val_df.to_csv(f'./Saves/LossFunction/m{c}.csv')
        objfn_val_df = objfn_val_df.fillna(objfn_val_df.min())
        
        dataframes.append(objfn_val_df)
        
    return dataframes

def plot_method_data(data, title='Instance Losses', dlabel='inst#', xlabel='Loss', ylabel='Iteration'):
    # create figure and axis
    fig, ax = plt.subplots()
    fig.set_figheight(4)
    fig.set_figwidth(8)
    
    # setting the axis' labels
    ax.set_ylabel(xlabel)
    ax.set_xlabel(ylabel)
    
    # Plot data
    for i in range(len(data)):
        data[i].T.plot(ax=ax, label=f'inst# {i}', figsize=(5, 3))
    ax.legend([f'{dlabel} {i}' for i in range(len(data))])
    plt.title(title)
    plt.show()  

def save_results(callback_results = None, objfn_val = None, accuracy_train = None, accuracy_test = None, weights = None):
    if callback_results:
        for c in range(len(callback_results)):
            objfn_val_df = pd.DataFrame([callback_results[c][i].objfun_vals for i in range(len(callback_results[c]))])
            objfn_val_df.to_csv(f'./Saves/LossFunction/m{c}.csv')

            weight_val = [callback_results[c][i].weight_vals for i in range(len(callback_results[c]))]

            for wr in range(len(weight_val)):
                weight_record = pd.DataFrame(weight_val[wr])
                weight_record.to_csv(f'./Saves/Weights/m{c}/sample_{wr}.csv')
    
    if objfn_val:
        for o in range(len(objfn_val)):
            objfn_val_df = pd.DataFrame(objfn_val[o][i] for i in range(len(objfn_val[o]))).astype('float')
            objfn_val_df.to_csv(f'./Saves/LossFunction/m{o}.csv')
    
    if accuracy_train:
        for a in range(len(accuracy_train)):
            accuracy_train_df = pd.DataFrame(accuracy_train[a][i] for i in range(len(accuracy_train[a])))
            accuracy_train_df.to_csv(f'./Saves/Scores/Train/m{a}.csv')

    if accuracy_test:
        for a in range(len(accuracy_test)):
            accuracy_test_df = pd.DataFrame(accuracy_test[a][i] for i in range(len(accuracy_test[a])))
            accuracy_test_df.to_csv(f'./Saves/Scores/Test/m{a}.csv')
    
    if weights:
        for w in range(len(weights)):
            for wr in range(len(weights[w])):
                weight_record = pd.DataFrame(weights[w][wr]).astype('float')
                weight_record.to_csv(f'./Saves/Weights/m{w}/sample_{wr}.csv')


def cross_validate(qnn, features, labels, K=5, loss='cross_entropy', maxiter=250):

    kf = KFold(n_splits=K, random_state=None)
    kfsplit = kf.split(features, labels)

    kf_score = []
    callback_results = []

    for k, (train, test) in enumerate(kfsplit):
        cf_callback_loop = classification_callback()

        classification_loop = NeuralNetworkClassifier(
            qnn,
            optimizer=COBYLA(maxiter),
            loss=loss,
            # one_hot=True,
            callback=cf_callback_loop.collect,
            warm_start=False
        )

        classification_loop.fit(features[train], labels[train])
        score = classification_loop.score(features[test], labels[test])
        kf_score.append(score)
        callback_results.append(classification_loop)
        print(f'Fold: {k+1}, Accuracy: {score}')

    return kf_score, callback_results
