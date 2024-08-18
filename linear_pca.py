import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import load_wine
from mpl_toolkits.mplot3d import Axes3D

# STEP1. Import data.
wine = load_wine()

fig = plt.figure(figsize = (12,5))
fig.subplots_adjust(wspace = 0.5)

colors = ['red', 'green', 'blue']

dimension = 12
X = wine.data[:,:dimension]
y = wine.target
feature_name = wine.feature_names[:dimension]
target_name = wine.target_names

# STEP2. Get the gram matrix.
S = np.cov(X, rowvar = 0, bias = 1)

# STEP3. Get the eigenvectors
eigen = np.linalg.eig(S)

print(eigen[0])

# STEP4. Project.
W = (eigen[1])[:,:2]
X_pca = X.dot(W)

ax = fig.add_subplot(1,1,1)
for yy in np.unique(y):
    ax.scatter(X_pca[:,0][y==yy],
                X_pca[:,1][y==yy],
                color=colors[yy],
                label=target_name[yy],
                edgecolors="black")

ax.set_xlabel("1st component")
ax.set_ylabel("2nd component")
ax.legend()
ax.set_title("2D Plot with PCA")

plt.show()
