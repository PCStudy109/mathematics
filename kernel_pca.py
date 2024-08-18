import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import load_wine
from mpl_toolkits.mplot3d import Axes3D
import itertools
import time

def rbf_innerproduct(a1, a2, sigma):
    ret = np.exp((-0.5) * np.square(np.linalg.norm(a1 - a2, ord = 2) / sigma))
    return ret

def rbf_total_product(X, sigma):
    v = 0.0
    l1 = range(0, X.shape[0])
    for i, j in itertools.product(l1, l1):
        v = 0
        v += rbf_innerproduct(X[i], X[j], sigma)
    return v

def rbf_oneside_product(X, idx, sigma):
    v = 0.0
    for i in range(0, X.shape[0]):
        v = 0.0
        v += rbf_innerproduct(X[idx], X[i], sigma)
    return v

def rbfpca_innerproduct(X, i, j, sigma, total_product):
    v = 0.0
    v += rbf_innerproduct(X[i], X[j], sigma)
    v -= rbf_oneside_product(X, i, sigma) / X.shape[0]
    v -= rbf_oneside_product(X, j, sigma) / X.shape[0]
    v += total_product
    return v

def rbfpca_gram(X, sigma):
    #print(X.shape[0])
    total_product = rbf_total_product(X, sigma) / (X.shape[0] * X.shape[0])
    gram = np.empty((0, X.shape[0]), float)
    for i in range(0, X.shape[0]):
        arr = np.zeros(0)
        for j in range(0, X.shape[0]):
            arr = np.append(arr, rbfpca_innerproduct(X, i, j, sigma, total_product))
        gram = np.append(gram, np.array([arr]), axis = 0)
    return gram

def rbfpca_projection(X, p, eigenvalues, eigenvectors):
    ev = eigenvectors.T
    T = np.empty((0, p), float)

    for i in range(0, X.shape[0]):
        arr = np.zeros(0)
        for j in range(0, p):
            arr = np.append(arr, np.sqrt(eigenvalues[j]) * ev[j][i])
        T = np.append(T, np.array([arr]), axis = 0)

    return T

# STEP1. Import Data.
wine = load_wine()

fig = plt.figure(figsize = (12,5))
fig.subplots_adjust(wspace = 0.5)

colors = ['red', 'green', 'blue']

dimension = 12
datanum = 178
sigma = 1.5

#print(wine)

X = wine.data[:datanum, :dimension]
y = wine.target[:datanum]
feature_name = wine.feature_names[:dimension]
target_name = wine.target_names

# STEP2. Get the gram matrix.
start = time.time()
G = rbfpca_gram(X, sigma)
#print(G)
#print(G.size)
end = time.time()
print('Get the gram matrix:' + str(end -start))

# STEP3. Get the eigenvectors.
start = time.time()
vals, vecs = np.linalg.eig(G)

idx = np.argsort(-vals)

vals = vals[idx]
vecs = vecs[:,idx]
#print('vals =\n', vals)
#print('vecs =\n', vecs)
end = time.time()

# STEP4. Perform the Projection.
start = time.time()
X_kpca = rbfpca_projection(X, 2, vals, vecs)
end = time.time()
#print(X_kpca)
print('Perform the Projection:' + str(end - start))

# STEP4. Plot.
ax2 = fig.add_subplot(1,1,1)
for yy in np.unique(y):
    ax2.scatter(X_kpca[:,0][y==yy],
                X_kpca[:,1][y==yy],
                color=colors[yy],
                label=target_name[yy],
                edgecolors="black")

ax2.set_xlabel("1st component")
ax2.set_ylabel("2nd component")
ax2.legend()
ax2.set_title("2D Plot with kPCA")

plt.show()
