import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import dendrogram, linkage

df = pd.DataFrame({
    "x": [1, 2, 3, 8, 9, 10],
    "y": [1, 2, 3, 8, 9, 10]
})

X = df.values
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

kmeans = KMeans(n_clusters=2)
labels_kmeans = kmeans.fit_predict(X_scaled)

plt.scatter(X_scaled[:,0], X_scaled[:,1], c=labels_kmeans)
plt.title("Clusters K-Means")
plt.show()

Z = linkage(X_scaled, method='ward')
plt.figure(figsize=(8,4))
dendrogram(Z)
plt.title("Dendrograma")
plt.show()

hier = AgglomerativeClustering(n_clusters=2, linkage='ward')
labels_hier = hier.fit_predict(X_scaled)

plt.scatter(X_scaled[:,0], X_scaled[:,1], c=labels_hier)
plt.title("Clusters Jerárquicos")
plt.show()

pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

plt.scatter(X_pca[:,0], X_pca[:,1], c=labels_kmeans)
plt.title("PCA 2D con K-Means")
plt.show()

pca3 = PCA(n_components=3)
X_pca3 = pca3.fit_transform(X_scaled)

from mpl_toolkits.mplot3d import Axes3D
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(X_pca3[:,0], X_pca3[:,1], X_pca3[:,2], c=labels_kmeans)
plt.title("PCA 3D con K-Means")
plt.show()
