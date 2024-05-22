#from qiskit_machine_learning.datasets import breast_cancer
from sklearn import datasets, utils
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from sklearn.preprocessing import OneHotEncoder, normalize, MinMaxScaler
import numpy as np
import pandas as pd

import GLOBAL_CONFIG

def cancer_data(PCA_n:int=GLOBAL_CONFIG.FEATURE_DIM, scale=True, split=True):
    X, y = datasets.load_breast_cancer(return_X_y=True)

    if PCA_n:
        pca = PCA(n_components=PCA_n)
        X = pca.fit_transform(X)
        

    if scale:
        scaler = MinMaxScaler(feature_range=(-1, 1))
        X = scaler.fit_transform(X)

    train_features, test_features, train_labels, test_labels = train_test_split(
        X, y, test_size=0.3, shuffle=True, random_state=42, stratify=y
    )

    # The training labels are in {0, 1}, we'll encode them {-1, 1}!
    train_labels = train_labels * 2 - 1
    test_labels = test_labels * 2 - 1

    print(f'Training set: {len(train_features)} samples')
    print(f'Testing set: {len(test_features)} samples')
    print(f'Number of features: {train_features.shape[-1]}')
    print(f'PCA Explained variance: {pca.explained_variance_}')
    print(f'Classes:{np.unique(y)}; Encoded as: {np.unique(train_labels)}')

    return train_features, test_features, train_labels, test_labels


def iris(pd=False, split=True):
    if pd == True:
        return sklearn_to_df(datasets.load_iris())

    data, target = datasets.load_iris(return_X_y=True)

    data = data[0: 99]
    target = target[0: 99]

    if split == False:
        target = target * 2 - 1
        print(f'Data set: {len(data)} samples')
        print(f'No train/test splitting')
        print(f'Number of features: {data.shape[-1]}')
        print(f'Classes: {np.unique(target)}')
        return data, target

    train_features, test_features, train_labels, test_labels = train_test_split(
        data,
        target,
        test_size=0.3,
        shuffle=True,
        random_state=42
    )

    # The training labels are in {0, 1}, we'll encode them {-1, 1}!
    train_labels = train_labels * 2 - 1
    test_labels = test_labels * 2 - 1

    # The training labels are in {0, 1, 2}, we'll one-hot encode them class labels!
    # training_labels_one_hot = OneHotEncoder(sparse_output=False).fit_transform(train_labels.reshape(-1,1))
    # test_labels_one_hot = OneHotEncoder(sparse_output=False).fit_transform(test_labels.reshape(-1,1))

    print(f'Training set: {len(train_features)} samples')
    print(f'Testing set: {len(test_features)} samples')
    print(f'Number of features: {train_features.shape[-1]}')
    print(f'Classes: {np.unique(train_labels)}')

    return train_features, test_features, train_labels, test_labels


def fetch_mnist(binary=True, PCA_n:int=GLOBAL_CONFIG.FEATURE_DIM, data_size=-1, scale=True, chars=['0', '1']):
    
    X, y = datasets.fetch_openml('mnist_784', version=1, return_X_y=True, parser='auto')

    if binary:
        data = pd.DataFrame(data=X)
        data['target'] = y
        data_binary = pd.concat(
            [
                data[data['target'] == chars[0]],
                data[data['target'] == chars[1]]
            ]
        )

        data_binary = utils.shuffle(data_binary, random_state=42)[:data_size]
        y = data_binary['target'].to_numpy().astype(int)
        X = data_binary.drop(columns=['target'])

    if PCA_n:
        pca = PCA(n_components=PCA_n)
        pca.fit(X)
        X = pca.transform(X)

    if scale:
        scaler = MinMaxScaler(feature_range=(-1, 1))
        X = scaler.fit_transform(X)

    train_features, test_features, train_labels, test_labels = train_test_split(
        X, y, test_size=0.3, shuffle=True, random_state=42
    )

    # The training labels are in {0, 1}, we'll encode them {-1, 1}!
    train_labels = train_labels * 2 - 1
    test_labels = test_labels * 2 - 1

    print(f'Training set: {len(train_features)} samples')
    print(f'Testing set: {len(test_features)} samples')
    print(f'Number of features: {train_features.shape[-1]}')
    print(f'Classes:{np.unique(y)}; Encoded as: {np.unique(train_labels)}')
    print(f'Training labels 1 ("{chars[1]}"): {train_labels[(train_labels[:] == 1)].shape[0]}, '+\
                         f'-1 ("{chars[0]}"): {train_labels[(train_labels[:] == -1)].shape[0]}')
    print(f'Test labels 1 ("{chars[1]}"): {test_labels[(test_labels[:] == 1)].shape[0]}, '+\
                     f'-1 ("{chars[0]}"): {test_labels[(test_labels[:] == -1)].shape[0]}')

    return train_features, test_features, train_labels, test_labels
    

def fetch_mnist_balanced(binary=True, PCA_n:int=GLOBAL_CONFIG.FEATURE_DIM, 
                         data_size=-1, scale=True, chars=['0', '1']):
    
    X, y = datasets.fetch_openml('mnist_784', version=1, return_X_y=True, parser='auto')

    if binary:
        data_size_0 = int(data_size / 2)
        data_size_1 = data_size - data_size_0
        data = pd.DataFrame(data=X)
        data['target'] = y
        data_binary_0 = data[data['target'] == chars[0]]
        data_binary_0 = utils.shuffle(data_binary_0, random_state=42)
        data_binary_0 = data_binary_0[:data_size_0]
        data_binary_1 = data[data['target'] == chars[1]]
        data_binary_1 = utils.shuffle(data_binary_1, random_state=42)
        data_binary_1 = data_binary_1[:data_size_1]
        
        data_binary = pd.concat([data_binary_0, data_binary_1])
        data_binary = utils.shuffle(data_binary, random_state=42)[:data_size]
        y = data_binary['target'].to_numpy().astype(int)
        X = data_binary.drop(columns=['target'])

    if PCA_n:
        pca = PCA(n_components=PCA_n)
        pca.fit(X)
        X = pca.transform(X)

    if scale:
        scaler = MinMaxScaler(feature_range=(-1, 1))
        X = scaler.fit_transform(X)

    train_features, test_features, train_labels, test_labels = train_test_split(
        X, y, test_size=0.3, shuffle=True, random_state=42, stratify=y
    )

    # The training labels are in {0, 1}, we'll encode them {-1, 1}!
    train_labels = train_labels * 2 - 1
    test_labels = test_labels * 2 - 1

    print(f'Training set: {len(train_features)} samples')
    print(f'Testing set: {len(test_features)} samples')
    print(f'Number of features: {train_features.shape[-1]}')
    print(f'Classes:{np.unique(y)}; Encoded as: {np.unique(train_labels)}')
    print(f'Training labels 1 ("{chars[1]}"): {train_labels[(train_labels[:] == 1)].shape[0]}, '+\
                         f'-1 ("{chars[0]}"): {train_labels[(train_labels[:] == -1)].shape[0]}')
    print(f'Test labels 1 ("{chars[1]}"): {test_labels[(test_labels[:] == 1)].shape[0]}, '+\
                     f'-1 ("{chars[0]}"): {test_labels[(test_labels[:] == -1)].shape[0]}')

    return train_features, test_features, train_labels, test_labels