import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib as mpl
import numpy as np
import seaborn as sns

train_data = pd.read_csv('./train.csv', sep=',')

train_data = train_data.sample(frac=1, random_state=42).reset_index(drop=True)
train_data = train_data.query('feature_4 > 24')
# Visualizing 5-D mix data using bubble charts
# leveraging the concepts of hue, size and depth
fig = plt.figure(figsize=(8, 6))
ax = fig.add_subplot(111, projection='3d')
t = fig.suptitle('Data Visualization', fontsize=14)

xs = list(train_data['feature_4'])
ys = list(train_data['feature_2'])
zs = list(train_data['feature_3'])
data_points = [(-x, y, z) for x, y, z in zip(xs, ys, zs)]
colors = list()
ss = list(train_data['feature_1'])
for wt in list(train_data['label']):
    if wt == 1:
        colors.append('red')
    elif wt == 0:
        colors.append('yellow')
    else:
        colors.append('blue')
# colors = ['red' if wt == 1 elif wt == 2 'yellow' for wt in list(train_data['label'])]

for data, color, size in zip(data_points, colors, ss):
    x, y, z = data
    ax.scatter(x, y, z, alpha=0.4, c=color, edgecolors='none', s=size)

ax.set_xlabel('Feature 4')
ax.set_ylabel('Feature 2')
ax.set_zlabel('Feature 3')
