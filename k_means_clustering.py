import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns


def import_data():
    df = pd.read_csv('Datasets/Iris.csv')
    df.drop(['Id', 'Species'], axis=1, inplace=True)
    return df


def plot_data(df, feature1, feature2):
    # plotting all samples
    sns.FacetGrid(df, height=6, aspect=0.8).map(plt.scatter, feature1, feature2).add_legend()
    plt.show()


# k-nearest neighbour implementation
def kmeans(df):
    k = int(input('\nEnter number of clusters: '))

    # choosing k clusters randomly
    centroids = df.sample(k)

    df_copy = df.copy()
    df_copy['Clusters'] = ''
    count = 0
    while True:
        cluster = {}
        for i in range(0, len(df)):
            dist = np.linalg.norm(centroids - df.iloc[i], axis=1)  # calculating distance of sample from each centroid
            cluster[i] = np.argmin(dist)  # gives index of centroid at min distance

        print(f'Iteration {count}: {cluster}')
        count = count + 1

        if list(df_copy['Clusters']) == list(cluster.values()):
            break
        df_copy['Clusters'] = cluster

        centroids = df_copy.groupby('Clusters').mean()


    sns.FacetGrid(df_copy, height=6, aspect=0.8, hue='Clusters').map(plt.scatter, 'SepalLengthCm', 'PetalLengthCm').add_legend()
    plt.show()


data = import_data()
plot_data(data, 'SepalLengthCm', 'PetalLengthCm')
kmeans(data)
