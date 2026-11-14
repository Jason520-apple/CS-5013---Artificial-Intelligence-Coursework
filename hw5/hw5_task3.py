#
# Template for Task 3: Kmeans Clustering
#
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

# --- Your Task --- #
# import libraries as needed 
from mpl_toolkits.mplot3d import Axes3D  #3D plots
# --- end of task --- #

# -------------------------------------
# load data 
# note we do not need label 
data = np.loadtxt('crimerate.csv', delimiter=',')
[n,p] = np.shape(data)
sample = data[:,0:-1]
# -------------------------------------

# --- Your Task --- #
# pick a proper number of clusters 
# k = 2 for fig 3, k = 3 for fig 4/5
# --- end of task --- #


# --- Your Task --- #
# implement the Kmeans clustering algorithm 
def kmeans_once(sample, k, max_iter=300, seed=0):
    """One run of Kmeans. Returns cluster labels (0..k-1) and the distortion."""
    rng = np.random.RandomState(seed)
    # you need to first randomly initialize k cluster centers
    centers = sample[rng.choice(len(sample), k, replace=False), :]
    label = np.zeros(len(sample), dtype=int)
    # then start a loop
    for it in range(max_iter):
        # assign each point to nearest center (Euclidean distance)
        dist = np.linalg.norm(sample[:, None, :] - centers[None, :, :], axis=2)
        new_label = np.argmin(dist, axis=1)
        # update each center as mean of points assigned to it
        new_centers = centers.copy()
        for j in range(k):
            if np.any(new_label == j):
                new_centers[j] = sample[new_label == j].mean(axis=0)
            else:  #empty cluster = re-seed with a random point
                new_centers[j] = sample[rng.randint(len(sample))]
        # stop when assignments no longer change (converged)
        if it > 0 and np.array_equal(new_label, label):
            break
        label, centers = new_label, new_centers
    # distortion = sum of squared distances from points to their own centers
    distortion = np.sum((sample - centers[label])**2)
    return label, distortion

def kmeans(sample, k, n_init=10):
    best_label, best_dist = None, np.inf
    for s in range(n_init):
        label, dist = kmeans_once(sample, k, seed=s)
        if dist < best_dist:
            best_label, best_dist = label, dist
    print('k=%d best distortion: %.3f' % (k, best_dist))
    return best_label

# when clustering is done, 
# store the clustering label in `label_cluster' 
# cluster index starts from 0 e.g., 
# label_cluster[0] = 1 means the 1st point assigned to cluster 1
# label_cluster[1] = 0 means the 2nd point assigned to cluster 0
# label_cluster[2] = 2 means the 3rd point assigned to cluster 2

colors = ['blue','red','green','m']

# PCA for visualization (feature reduction, as given in the template)
pca2 = PCA(n_components=2).fit(sample)
sample_pca2 = pca2.transform(sample)
pca3 = PCA(n_components=3).fit(sample)
sample_pca3 = pca3.transform(sample)

def plot2d(label_cluster, k, title):
    plt.figure()
    for i in range(k):
        idx = np.where(label_cluster == i)[0]
        plt.scatter(sample_pca2[idx,0], sample_pca2[idx,1], edgecolors=colors[i], facecolors='none', label='Cluster %d' % i)
    plt.xlabel('PC 1'); plt.ylabel('PC 2'); plt.title(title); plt.legend()

# --- end of task --- #

# Figure 3: k = 2, 2D
k = 2
label_cluster = kmeans(sample, k)
print('k=2 cluster sizes:', np.bincount(label_cluster))
plot2d(label_cluster, k, 'Figure 3: Kmeans (k=2), 2D')

# Figure 4: k = 3, 2D
k = 3
label_cluster = kmeans(sample, k)
print('k=3 cluster sizes:', np.bincount(label_cluster))
plot2d(label_cluster, k, 'Figure 4: Kmeans (k=3), 2D')

# Figure 5: k = 3, 3D
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
for i in range(k):
    idx = np.where(label_cluster == i)[0]
    ax.scatter(sample_pca3[idx,0], sample_pca3[idx,1], sample_pca3[idx,2], edgecolors=colors[i], facecolors='none', label='Cluster %d' % i)
ax.set_xlabel('PC 1'); ax.set_ylabel('PC 2'); ax.set_zlabel('PC 3')
ax.set_title('Figure 5: Kmeans (k=3), 3D'); ax.legend()
plt.show()