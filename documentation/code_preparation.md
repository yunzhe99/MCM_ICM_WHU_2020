我一直在做的事情主要是机器学习，虽然和计算数学做的差不多的事情，但是我们可能用到的方法更新一点。我查阅了一些数模的书籍和资料，发现他们的方法都很老，但是在特定的方法上也能取得比较好的效果。

数模竞赛和正常的科研还是有所区别，因为更注重速度，而不是效果。因此还是准备了一些经典方法，以期能够直接调用。

代码主要基于sklearn编写。

# 有监督学习

有监督学习是指有数据有标签的学习。

## K-近邻

最近邻方法背后的原理是从训练样本中找到与新点在距离上最近的预定数量的几个点，然后从这些点中预测标签。 

最基本的代码(对鸢尾花数据集分类)：

```python
from sklearn.datasets import load_iris
from sklearn import neighbors
import sklearn
 
#查看iris数据集
iris = load_iris()
print iris
 
knn = neighbors.KNeighborsClassifier()
#训练数据集
knn.fit(iris.data, iris.target)
#预测
predict = knn.predict([[0.1,0.2,0.3,0.4]])
print predict
print iris.target_names[predict]</span>
```

参考文献：https://sklearn.apachecn.org/docs/0.21.3/7.html

## 线性回归

目标值 y 是输入变量 x 的线性组合，当然可以有很多变种。

```python
import matplotlib.pyplot as plt
import numpy as np
from sklearn import datasets, linear_model
from sklearn.metrics import mean_squared_error, r2_score

# Load the diabetes dataset
diabetes_X, diabetes_y = datasets.load_diabetes(return_X_y=True)

# Use only one feature
diabetes_X = diabetes_X[:, np.newaxis, 2]

# Split the data into training/testing sets
diabetes_X_train = diabetes_X[:-20]
diabetes_X_test = diabetes_X[-20:]

# Split the targets into training/testing sets
diabetes_y_train = diabetes_y[:-20]
diabetes_y_test = diabetes_y[-20:]

# Create linear regression object
regr = linear_model.LinearRegression()

# Train the model using the training sets
regr.fit(diabetes_X_train, diabetes_y_train)

# Make predictions using the testing set
diabetes_y_pred = regr.predict(diabetes_X_test)

# The coefficients
print('Coefficients: \n', regr.coef_)
# The mean squared error
print('Mean squared error: %.2f'
      % mean_squared_error(diabetes_y_test, diabetes_y_pred))
# The coefficient of determination: 1 is perfect prediction
print('Coefficient of determination: %.2f'
      % r2_score(diabetes_y_test, diabetes_y_pred))

# Plot outputs
plt.scatter(diabetes_X_test, diabetes_y_test,  color='black')
plt.plot(diabetes_X_test, diabetes_y_pred, color='blue', linewidth=3)

plt.xticks(())
plt.yticks(())

plt.show()
```

参考文献：https://sklearn.apachecn.org/docs/0.21.3/2.html

## 逻辑回归

logistic 回归，虽然名字里有 “回归” 二字，但实际上是解决分类问题的一类线性模型。在某些文献中，logistic 回归又被称作 logit 回归，maximum-entropy classification（MaxEnt，最大熵分类），或 log-linear classifier（对数线性分类器）。该模型利用函数 [logistic function](https://en.wikipedia.org/wiki/Logistic_function) 将单次试验（single trial）的可能结果输出为概率。

```python
import time
import matplotlib.pyplot as plt
import numpy as np

from sklearn.datasets import fetch_openml
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.utils import check_random_state

print(__doc__)

# Author: Arthur Mensch <arthur.mensch@m4x.org>
# License: BSD 3 clause

# Turn down for faster convergence
t0 = time.time()
train_samples = 5000

# Load data from https://www.openml.org/d/554
X, y = fetch_openml('mnist_784', version=1, return_X_y=True)

random_state = check_random_state(0)
permutation = random_state.permutation(X.shape[0])
X = X[permutation]
y = y[permutation]
X = X.reshape((X.shape[0], -1))

X_train, X_test, y_train, y_test = train_test_split(
    X, y, train_size=train_samples, test_size=10000)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Turn up tolerance for faster convergence
clf = LogisticRegression(
    C=50. / train_samples, penalty='l1', solver='saga', tol=0.1
)
clf.fit(X_train, y_train)
sparsity = np.mean(clf.coef_ == 0) * 100
score = clf.score(X_test, y_test)
# print('Best C % .4f' % clf.C_)
print("Sparsity with L1 penalty: %.2f%%" % sparsity)
print("Test score with L1 penalty: %.4f" % score)

coef = clf.coef_.copy()
plt.figure(figsize=(10, 5))
scale = np.abs(coef).max()
for i in range(10):
    l1_plot = plt.subplot(2, 5, i + 1)
    l1_plot.imshow(coef[i].reshape(28, 28), interpolation='nearest',
                   cmap=plt.cm.RdBu, vmin=-scale, vmax=scale)
    l1_plot.set_xticks(())
    l1_plot.set_yticks(())
    l1_plot.set_xlabel('Class %i' % i)
plt.suptitle('Classification vector for...')

run_time = time.time() - t0
print('Example run in %.3f s' % run_time)
plt.show()
```

参考文献：[https://sklearn.apachecn.org/docs/0.21.3/2.html#1111-logistic-%E5%9B%9E%E5%BD%92](https://sklearn.apachecn.org/docs/0.21.3/2.html#1111-logistic-回归)

## 支持向量机

支持向量机的优势在于:

- 在高维空间中非常高效.
- 即使在数据维度比样本数量大的情况下仍然有效.
- 在决策函数（称为支持向量）中使用训练集的子集,因此它也是高效利用内存的.
- 通用性: 不同的核函数 [核函数](https://sklearn.apachecn.org/docs/0.21.3/5.html#146-核函数) 与特定的决策函数一一对应.常见的 kernel 已经提供,也可以指定定制的内核.

```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn import svm
from sklearn.datasets import make_blobs


# we create 40 separable points
X, y = make_blobs(n_samples=40, centers=2, random_state=6)

# fit the model, don't regularize for illustration purposes
clf = svm.SVC(kernel='linear', C=1000)
clf.fit(X, y)

plt.scatter(X[:, 0], X[:, 1], c=y, s=30, cmap=plt.cm.Paired)

# plot the decision function
ax = plt.gca()
xlim = ax.get_xlim()
ylim = ax.get_ylim()

# create grid to evaluate model
xx = np.linspace(xlim[0], xlim[1], 30)
yy = np.linspace(ylim[0], ylim[1], 30)
YY, XX = np.meshgrid(yy, xx)
xy = np.vstack([XX.ravel(), YY.ravel()]).T
Z = clf.decision_function(xy).reshape(XX.shape)

# plot decision boundary and margins
ax.contour(XX, YY, Z, colors='k', levels=[-1, 0, 1], alpha=0.5,
           linestyles=['--', '-', '--'])
# plot support vectors
ax.scatter(clf.support_vectors_[:, 0], clf.support_vectors_[:, 1], s=100,
           linewidth=1, facecolors='none', edgecolors='k')
plt.show()
```

参考文献：https://sklearn.apachecn.org/docs/0.21.3/5.html

## 决策树与随机森林

**Decision Trees (DTs)** 是一种用来 [classification](https://sklearn.apachecn.org/docs/0.21.3/11.html#1101-分类) 和 [regression](https://sklearn.apachecn.org/docs/0.21.3/11.html#1102-回归) 的无参监督学习方法。其目的是创建一种模型从数据特征中学习简单的决策规则来预测一个目标变量的值。

```python
# Import the necessary modules and libraries
import numpy as np
from sklearn.tree import DecisionTreeRegressor
import matplotlib.pyplot as plt

# Create a random dataset
rng = np.random.RandomState(1)
X = np.sort(5 * rng.rand(80, 1), axis=0)
y = np.sin(X).ravel()
y[::5] += 3 * (0.5 - rng.rand(16))

# Fit regression model
regr_1 = DecisionTreeRegressor(max_depth=2)
regr_2 = DecisionTreeRegressor(max_depth=5)
regr_1.fit(X, y)
regr_2.fit(X, y)

# Predict
X_test = np.arange(0.0, 5.0, 0.01)[:, np.newaxis]
y_1 = regr_1.predict(X_test)
y_2 = regr_2.predict(X_test)

# Plot the results
plt.figure()
plt.scatter(X, y, s=20, edgecolor="black",
            c="darkorange", label="data")
plt.plot(X_test, y_1, color="cornflowerblue",
         label="max_depth=2", linewidth=2)
plt.plot(X_test, y_2, color="yellowgreen", label="max_depth=5", linewidth=2)
plt.xlabel("data")
plt.ylabel("target")
plt.title("Decision Tree Regression")
plt.legend()
plt.show()
```

## 神经网络

**多层感知器(MLP)** 是一种监督学习算法。

```python
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_openml
from sklearn.neural_network import MLPClassifier

print(__doc__)

# Load data from https://www.openml.org/d/554
X, y = fetch_openml('mnist_784', version=1, return_X_y=True)
X = X / 255.

# rescale the data, use the traditional train/test split
X_train, X_test = X[:60000], X[60000:]
y_train, y_test = y[:60000], y[60000:]

mlp = MLPClassifier(hidden_layer_sizes=(50,), max_iter=10, alpha=1e-4,
                    solver='sgd', verbose=10, random_state=1,
                    learning_rate_init=.1)

mlp.fit(X_train, y_train)
print("Training set score: %f" % mlp.score(X_train, y_train))
print("Test set score: %f" % mlp.score(X_test, y_test))

fig, axes = plt.subplots(4, 4)
# use global min / max to ensure all weights are shown on the same scale
vmin, vmax = mlp.coefs_[0].min(), mlp.coefs_[0].max()
for coef, ax in zip(mlp.coefs_[0].T, axes.ravel()):
    ax.matshow(coef.reshape(28, 28), cmap=plt.cm.gray, vmin=.5 * vmin,
               vmax=.5 * vmax)
    ax.set_xticks(())
    ax.set_yticks(())

plt.show()
```

# 无监督学习

无监督学习的数据是未经标记的。

## 聚类算法

聚类就是把一些数据自动分类。

### k-平均算法

[`KMeans`](https://scikit-learn.org/stable/modules/generated/sklearn.cluster.KMeans.html#sklearn.cluster.KMeans) 算法通过把样本分离成 n 个具有相同方差的类的方式来聚集数据，最小化称为 惯量([inertia](https://sklearn.apachecn.org/docs/0.21.3/inertia)) 或 簇内平方和(within-cluster sum-of-squares)的标准（criterion）。该算法需要指定簇的数量。它可以很好地扩展到大量样本(large number of samples)，并已经被广泛应用于许多不同领域的应用领域。

```python
from time import time
import numpy as np
import matplotlib.pyplot as plt

from sklearn import metrics
from sklearn.cluster import KMeans
from sklearn.datasets import load_digits
from sklearn.decomposition import PCA
from sklearn.preprocessing import scale

np.random.seed(42)

X_digits, y_digits = load_digits(return_X_y=True)
data = scale(X_digits)

n_samples, n_features = data.shape
n_digits = len(np.unique(y_digits))
labels = y_digits

sample_size = 300

print("n_digits: %d, \t n_samples %d, \t n_features %d"
      % (n_digits, n_samples, n_features))


print(82 * '_')
print('init\t\ttime\tinertia\thomo\tcompl\tv-meas\tARI\tAMI\tsilhouette')


def bench_k_means(estimator, name, data):
    t0 = time()
    estimator.fit(data)
    print('%-9s\t%.2fs\t%i\t%.3f\t%.3f\t%.3f\t%.3f\t%.3f\t%.3f'
          % (name, (time() - t0), estimator.inertia_,
             metrics.homogeneity_score(labels, estimator.labels_),
             metrics.completeness_score(labels, estimator.labels_),
             metrics.v_measure_score(labels, estimator.labels_),
             metrics.adjusted_rand_score(labels, estimator.labels_),
             metrics.adjusted_mutual_info_score(labels,  estimator.labels_),
             metrics.silhouette_score(data, estimator.labels_,
                                      metric='euclidean',
                                      sample_size=sample_size)))

bench_k_means(KMeans(init='k-means++', n_clusters=n_digits, n_init=10),
              name="k-means++", data=data)

bench_k_means(KMeans(init='random', n_clusters=n_digits, n_init=10),
              name="random", data=data)

# in this case the seeding of the centers is deterministic, hence we run the
# kmeans algorithm only once with n_init=1
pca = PCA(n_components=n_digits).fit(data)
bench_k_means(KMeans(init=pca.components_, n_clusters=n_digits, n_init=1),
              name="PCA-based",
              data=data)
print(82 * '_')

# #############################################################################
# Visualize the results on PCA-reduced data

reduced_data = PCA(n_components=2).fit_transform(data)
kmeans = KMeans(init='k-means++', n_clusters=n_digits, n_init=10)
kmeans.fit(reduced_data)

# Step size of the mesh. Decrease to increase the quality of the VQ.
h = .02     # point in the mesh [x_min, x_max]x[y_min, y_max].

# Plot the decision boundary. For that, we will assign a color to each
x_min, x_max = reduced_data[:, 0].min() - 1, reduced_data[:, 0].max() + 1
y_min, y_max = reduced_data[:, 1].min() - 1, reduced_data[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))

# Obtain labels for each point in mesh. Use last trained model.
Z = kmeans.predict(np.c_[xx.ravel(), yy.ravel()])

# Put the result into a color plot
Z = Z.reshape(xx.shape)
plt.figure(1)
plt.clf()
plt.imshow(Z, interpolation='nearest',
           extent=(xx.min(), xx.max(), yy.min(), yy.max()),
           cmap=plt.cm.Paired,
           aspect='auto', origin='lower')

plt.plot(reduced_data[:, 0], reduced_data[:, 1], 'k.', markersize=2)
# Plot the centroids as a white X
centroids = kmeans.cluster_centers_
plt.scatter(centroids[:, 0], centroids[:, 1],
            marker='x', s=169, linewidths=3,
            color='w', zorder=10)
plt.title('K-means clustering on the digits dataset (PCA-reduced data)\n'
          'Centroids are marked with white cross')
plt.xlim(x_min, x_max)
plt.ylim(y_min, y_max)
plt.xticks(())
plt.yticks(())
plt.show()
```

### 层次聚类分析

层次聚类(Hierarchical clustering)代表着一类的聚类算法，这种类别的算法通过不断的合并或者分割内置聚类来构建最终聚类。 聚类的层次可以被表示成树（或者树形图(dendrogram)）。树根是拥有所有样本的唯一聚类，叶子是仅有一个样本的聚类。 请参照 [Wikipedia page](https://en.wikipedia.org/wiki/Hierarchical_clustering) 查看更多细节。

```python
from time import time

import numpy as np
from scipy import ndimage
from matplotlib import pyplot as plt

from sklearn import manifold, datasets

X, y = datasets.load_digits(return_X_y=True)
n_samples, n_features = X.shape

np.random.seed(0)

def nudge_images(X, y):
    # Having a larger dataset shows more clearly the behavior of the
    # methods, but we multiply the size of the dataset only by 2, as the
    # cost of the hierarchical clustering methods are strongly
    # super-linear in n_samples
    shift = lambda x: ndimage.shift(x.reshape((8, 8)),
                                  .3 * np.random.normal(size=2),
                                  mode='constant',
                                  ).ravel()
    X = np.concatenate([X, np.apply_along_axis(shift, 1, X)])
    Y = np.concatenate([y, y], axis=0)
    return X, Y


X, y = nudge_images(X, y)


#----------------------------------------------------------------------
# Visualize the clustering
def plot_clustering(X_red, labels, title=None):
    x_min, x_max = np.min(X_red, axis=0), np.max(X_red, axis=0)
    X_red = (X_red - x_min) / (x_max - x_min)

    plt.figure(figsize=(6, 4))
    for i in range(X_red.shape[0]):
        plt.text(X_red[i, 0], X_red[i, 1], str(y[i]),
                 color=plt.cm.nipy_spectral(labels[i] / 10.),
                 fontdict={'weight': 'bold', 'size': 9})

    plt.xticks([])
    plt.yticks([])
    if title is not None:
        plt.title(title, size=17)
    plt.axis('off')
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])

#----------------------------------------------------------------------
# 2D embedding of the digits dataset
print("Computing embedding")
X_red = manifold.SpectralEmbedding(n_components=2).fit_transform(X)
print("Done.")

from sklearn.cluster import AgglomerativeClustering

for linkage in ('ward', 'average', 'complete', 'single'):
    clustering = AgglomerativeClustering(linkage=linkage, n_clusters=10)
    t0 = time()
    clustering.fit(X_red)
    print("%s :\t%.2fs" % (linkage, time() - t0))

    plot_clustering(X_red, clustering.labels_, "%s linkage" % linkage)


plt.show()
```

参考文献：https://sklearn.apachecn.org/docs/0.21.3/22.html

### 最大期望算法

 [期望最大化（Expectation-maximization，EM）](https://en.wikipedia.org/wiki/Expectation–maximization_algorithm) 是一个理论完善的统计算法，其通过迭代方式来解决这个问题。首先，假设一个随机分量 （随机地选择一个中心点，该点可以由 k-means 算法得到，或者甚至可以在原点周围随意选取一个点）,并为每个点分别计算由该混合模型内的每个分量生成的概率。然后，调整模型参数以最大化模型生成这些参数的可能性。重复这个过程，该算法保证该过程内的参数总会收敛到一个局部最优解。

```python
import itertools

import numpy as np
from scipy import linalg
import matplotlib.pyplot as plt
import matplotlib as mpl

from sklearn import mixture

print(__doc__)

color_iter = itertools.cycle(['navy', 'c', 'cornflowerblue', 'gold',
                              'darkorange'])


def plot_results(X, Y, means, covariances, index, title):
    splot = plt.subplot(5, 1, 1 + index)
    for i, (mean, covar, color) in enumerate(zip(
            means, covariances, color_iter)):
        v, w = linalg.eigh(covar)
        v = 2. * np.sqrt(2.) * np.sqrt(v)
        u = w[0] / linalg.norm(w[0])
        # as the DP will not use every component it has access to
        # unless it needs it, we shouldn't plot the redundant
        # components.
        if not np.any(Y == i):
            continue
        plt.scatter(X[Y == i, 0], X[Y == i, 1], .8, color=color)

        # Plot an ellipse to show the Gaussian component
        angle = np.arctan(u[1] / u[0])
        angle = 180. * angle / np.pi  # convert to degrees
        ell = mpl.patches.Ellipse(mean, v[0], v[1], 180. + angle, color=color)
        ell.set_clip_box(splot.bbox)
        ell.set_alpha(0.5)
        splot.add_artist(ell)

    plt.xlim(-6., 4. * np.pi - 6.)
    plt.ylim(-5., 5.)
    plt.title(title)
    plt.xticks(())
    plt.yticks(())


def plot_samples(X, Y, n_components, index, title):
    plt.subplot(5, 1, 4 + index)
    for i, color in zip(range(n_components), color_iter):
        # as the DP will not use every component it has access to
        # unless it needs it, we shouldn't plot the redundant
        # components.
        if not np.any(Y == i):
            continue
        plt.scatter(X[Y == i, 0], X[Y == i, 1], .8, color=color)

    plt.xlim(-6., 4. * np.pi - 6.)
    plt.ylim(-5., 5.)
    plt.title(title)
    plt.xticks(())
    plt.yticks(())


# Parameters
n_samples = 100

# Generate random sample following a sine curve
np.random.seed(0)
X = np.zeros((n_samples, 2))
step = 4. * np.pi / n_samples

for i in range(X.shape[0]):
    x = i * step - 6.
    X[i, 0] = x + np.random.normal(0, 0.1)
    X[i, 1] = 3. * (np.sin(x) + np.random.normal(0, .2))

plt.figure(figsize=(10, 10))
plt.subplots_adjust(bottom=.04, top=0.95, hspace=.2, wspace=.05,
                    left=.03, right=.97)

# Fit a Gaussian mixture with EM using ten components
gmm = mixture.GaussianMixture(n_components=10, covariance_type='full',
                              max_iter=100).fit(X)
plot_results(X, gmm.predict(X), gmm.means_, gmm.covariances_, 0,
             'Expectation-maximization')

dpgmm = mixture.BayesianGaussianMixture(
    n_components=10, covariance_type='full', weight_concentration_prior=1e-2,
    weight_concentration_prior_type='dirichlet_process',
    mean_precision_prior=1e-2, covariance_prior=1e0 * np.eye(2),
    init_params="random", max_iter=100, random_state=2).fit(X)
plot_results(X, dpgmm.predict(X), dpgmm.means_, dpgmm.covariances_, 1,
             "Bayesian Gaussian mixture models with a Dirichlet process prior "
             r"for $\gamma_0=0.01$.")

X_s, y_s = dpgmm.sample(n_samples=2000)
plot_samples(X_s, y_s, dpgmm.n_components, 0,
             "Gaussian mixture with a Dirichlet process prior "
             r"for $\gamma_0=0.01$ sampled with $2000$ samples.")

dpgmm = mixture.BayesianGaussianMixture(
    n_components=10, covariance_type='full', weight_concentration_prior=1e+2,
    weight_concentration_prior_type='dirichlet_process',
    mean_precision_prior=1e-2, covariance_prior=1e0 * np.eye(2),
    init_params="kmeans", max_iter=100, random_state=2).fit(X)
plot_results(X, dpgmm.predict(X), dpgmm.means_, dpgmm.covariances_, 2,
             "Bayesian Gaussian mixture models with a Dirichlet process prior "
             r"for $\gamma_0=100$")

X_s, y_s = dpgmm.sample(n_samples=2000)
plot_samples(X_s, y_s, dpgmm.n_components, 1,
             "Gaussian mixture with a Dirichlet process prior "
             r"for $\gamma_0=100$ sampled with $2000$ samples.")

plt.show()
```

参考文献：https://sklearn.apachecn.org/docs/0.21.3/20.html

## 可视化与降维

这一点对美赛极为重要。

### 主成分分析

[`decomposition.PCA`](https://scikit-learn.org/stable/modules/generated/sklearn.decomposition.PCA.html#sklearn.decomposition.PCA) 寻找能够捕捉原始特征的差异的特征的组合. 请参阅 [分解成分中的信号（矩阵分解问题）](https://sklearn.apachecn.org/docs/0.21.3/24#25-分解成分中的信号（矩阵分解问题）).

```python
from time import time
import logging
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.datasets import fetch_lfw_people
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.decomposition import PCA
from sklearn.svm import SVC


print(__doc__)

# Display progress logs on stdout
logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s')


# #############################################################################
# Download the data, if not already on disk and load it as numpy arrays

lfw_people = fetch_lfw_people(min_faces_per_person=70, resize=0.4)

# introspect the images arrays to find the shapes (for plotting)
n_samples, h, w = lfw_people.images.shape

# for machine learning we use the 2 data directly (as relative pixel
# positions info is ignored by this model)
X = lfw_people.data
n_features = X.shape[1]

# the label to predict is the id of the person
y = lfw_people.target
target_names = lfw_people.target_names
n_classes = target_names.shape[0]

print("Total dataset size:")
print("n_samples: %d" % n_samples)
print("n_features: %d" % n_features)
print("n_classes: %d" % n_classes)


# #############################################################################
# Split into a training set and a test set using a stratified k fold

# split into a training and testing set
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42)


# #############################################################################
# Compute a PCA (eigenfaces) on the face dataset (treated as unlabeled
# dataset): unsupervised feature extraction / dimensionality reduction
n_components = 150

print("Extracting the top %d eigenfaces from %d faces"
      % (n_components, X_train.shape[0]))
t0 = time()
pca = PCA(n_components=n_components, svd_solver='randomized',
          whiten=True).fit(X_train)
print("done in %0.3fs" % (time() - t0))

eigenfaces = pca.components_.reshape((n_components, h, w))

print("Projecting the input data on the eigenfaces orthonormal basis")
t0 = time()
X_train_pca = pca.transform(X_train)
X_test_pca = pca.transform(X_test)
print("done in %0.3fs" % (time() - t0))


# #############################################################################
# Train a SVM classification model

print("Fitting the classifier to the training set")
t0 = time()
param_grid = {'C': [1e3, 5e3, 1e4, 5e4, 1e5],
              'gamma': [0.0001, 0.0005, 0.001, 0.005, 0.01, 0.1], }
clf = GridSearchCV(
    SVC(kernel='rbf', class_weight='balanced'), param_grid
)
clf = clf.fit(X_train_pca, y_train)
print("done in %0.3fs" % (time() - t0))
print("Best estimator found by grid search:")
print(clf.best_estimator_)


# #############################################################################
# Quantitative evaluation of the model quality on the test set

print("Predicting people's names on the test set")
t0 = time()
y_pred = clf.predict(X_test_pca)
print("done in %0.3fs" % (time() - t0))

print(classification_report(y_test, y_pred, target_names=target_names))
print(confusion_matrix(y_test, y_pred, labels=range(n_classes)))


# #############################################################################
# Qualitative evaluation of the predictions using matplotlib

def plot_gallery(images, titles, h, w, n_row=3, n_col=4):
    """Helper function to plot a gallery of portraits"""
    plt.figure(figsize=(1.8 * n_col, 2.4 * n_row))
    plt.subplots_adjust(bottom=0, left=.01, right=.99, top=.90, hspace=.35)
    for i in range(n_row * n_col):
        plt.subplot(n_row, n_col, i + 1)
        plt.imshow(images[i].reshape((h, w)), cmap=plt.cm.gray)
        plt.title(titles[i], size=12)
        plt.xticks(())
        plt.yticks(())


# plot the result of the prediction on a portion of the test set

def title(y_pred, y_test, target_names, i):
    pred_name = target_names[y_pred[i]].rsplit(' ', 1)[-1]
    true_name = target_names[y_test[i]].rsplit(' ', 1)[-1]
    return 'predicted: %s\ntrue:      %s' % (pred_name, true_name)

prediction_titles = [title(y_pred, y_test, target_names, i)
                     for i in range(y_pred.shape[0])]

plot_gallery(X_test, prediction_titles, h, w)

# plot the gallery of the most significative eigenfaces

eigenface_titles = ["eigenface %d" % i for i in range(eigenfaces.shape[0])]
plot_gallery(eigenfaces, eigenface_titles, h, w)

plt.show()
```

参考文献：https://sklearn.apachecn.org/docs/0.21.3/42.html

### 核主成分分析

和PCA手感差不多，细节有所区别。

```python
from sklearn.datasets import load_digits
from sklearn.decomposition import KernelPCA
X, _ = load_digits(return_X_y=True)
transformer = KernelPCA(n_components=7, kernel='linear')
X_transformed = transformer.fit_transform(X)
X_transformed.shape
```

参考文献：https://scikit-learn.org/stable/modules/generated/sklearn.decomposition.KernelPCA.html

# 仿生算法

仿生算法不知道为什么在美赛中用得特别多，好像还蛮吃香的。变种很多，挑几个典型的写一下。

## 遗传算法

遗传算法通常实现方式为一种[计算机模拟](https://zh.wikipedia.org/wiki/计算机模拟)。对于一个最优化问题，一定数量的[候选解](https://zh.wikipedia.org/w/index.php?title=候选解&action=edit&redlink=1)（称为个体）可抽象表示为[染色体](https://zh.wikipedia.org/wiki/染色體_(遺傳演算法))，使[种群](https://zh.wikipedia.org/wiki/种群)向更好的解进化。传统上，解用[二进制](https://zh.wikipedia.org/wiki/二进制)表示（即0和1的串），但也可以用其他表示方法。进化从完全[随机](https://zh.wikipedia.org/wiki/随机)个体的种群开始，之后一代一代发生。在每一代中评价整个种群的[适应度](https://zh.wikipedia.org/wiki/适应度)，从当前种群中随机地选择多个个体（基于它们的适应度），通过自然选择和突变产生新的生命种群，该种群在算法的下一次迭代中成为当前种群。

```python
from GA import GA_TSP
import numpy as np

num_points = 8

points = range(num_points)
points_coordinate = np.random.rand(num_points, 2)
distance_matrix = np.zeros(shape=(num_points, num_points))
for i in range(num_points):
    for j in range(num_points):
        distance_matrix[i][j] = np.linalg.norm(points_coordinate[i] - points_coordinate[j], ord=2)
print('distance_matrix is: \n', distance_matrix)


def cal_total_distance(points):
    num_points, = points.shape
    total_distance = 0
    for i in range(num_points - 1):
        total_distance += distance_matrix[points[i], points[i + 1]]
    total_distance += distance_matrix[points[i + 1], points[0]]
    return total_distance

```

## 免疫算法

和遗传算法类似，不过是模仿人类免疫系统。

美赛用得少，可能是因为代码不太好写。

```python
import numpy as np
import ObjFunction


class AIAIndividual:

    '''
    individual of artificial immune algorithm
    '''

    def __init__(self,  vardim, bound):
        '''
        vardim: dimension of variables
        bound: boundaries of variables
        '''
        self.vardim = vardim
        self.bound = bound
        self.fitness = 0.
        self.trials = 0
        self.concentration = 0

    def generate(self):
        '''
        generate a random chromsome for artificial immune algorithm
        '''
        len = self.vardim
        rnd = np.random.random(size=len)
        self.chrom = np.zeros(len)
        for i in xrange(0, len):
            self.chrom[i] = self.bound[0, i] + \
                (self.bound[1, i] - self.bound[0, i]) * rnd[i]

    def calculateFitness(self):
        '''
        calculate the fitness of the chromsome
        '''
        self.fitness = ObjFunction.GrieFunc(
            self.vardim, self.chrom, self.bound)
            
            
import numpy as np
from AIAIndividual import AIAIndividual
import random
import copy
import matplotlib.pyplot as plt


class ArtificialImmuneAlgorithm:

    '''
    The class for artificial immune algorithm
    '''

    def __init__(self, sizepop, sizemem, vardim, bound, MAXGEN, params):
        '''
        sizepop: population sizepop
        vardim: dimension of variables
        bound: boundaries of variables
        MAXGEN: termination condition
        params: algorithm required parameters, it is a list which is consisting of [mutation rate, cloneNum]
        '''
        self.sizepop = sizepop
        self.sizemem = sizemem
        self.MAXGEN = MAXGEN
        self.vardim = vardim
        self.bound = bound
        self.population = []
        self.clonePopulation = []
        self.memories = []
        self.cloneMemories = []
        self.popFitness = np.zeros(self.sizepop)
        self.popCloneFitness = np.zeros(
            int(self.sizepop * self.sizepop * params[1]))
        self.memfitness = np.zero(self.sizemem)
        self.memClonefitness = np.zero(
            int(self.sizemem * self.sizemem * params[1]))
        self.trace = np.zeros((self.MAXGEN, 2))
        self.params = params

    def initialize(self):
        '''
        initialize the population
        '''
        for i in xrange(0, self.sizepop):
            ind = AIAIndividual(self.vardim, self.bound)
            ind.generate()
            self.population.append(ind)
        for i in xrange(0, self.sizemem):
            ind = AIAIndividual(self.vardim, self.bound)
            ind.generate()
            self.memories.append(ind)

    def evaluatePopulation(self, flag):
        '''
        evaluation of the population fitnesses
        '''
        if flag == 1:
            for i in xrange(0, self.sizepop):
                self.population[i].calculateFitness()
                self.popFitness[i] = self.population[i].fitness
        else:
            for i in xrange(0, self.sizemem):
                self.memories[i].calculateFitness()
                self.memfitness[i] = self.memories[i].fitness

    def evaluateClone(self, flag):
        '''
        evaluation of the clone fitnesses
        '''
        if flag == 1:
            for i in xrange(0, self.sizepop):
                self.clonePopulation[i].calculateFitness()
                self.popCloneFitness[i] = self.clonePopulation[i].fitness
        else:
            for i in xrange(0, self.sizemem):
                self.cloneMemories[i].calculateFitness()
                self.memClonefitness[i] = self.cloneMemories[i].fitness

    def solve(self):
        '''
        evolution process of artificial immune algorithm
        '''
        self.t = 0
        self.initialize()
        self.best = AIAIndividual(self.vardim, self.bound)
        while (self.t < self.MAXGEN):
            # evolution of population
            self.cloneOperation(1)
            self.mutationOperation(1)
            self.evaluatePopulation(1)
            self.selectionOperation(1)

            # evolution of memories
            self.cloneOperation(2)
            self.mutationOperation(2)
            self.evaluatePopulation()
            self.selectionOperation(2)

            best = np.max(self.popFitness)
            bestIndex = np.argmax(self.popFitness)
            if best > self.best.fitness:
                self.best = copy.deepcopy(self.population[bestIndex])
            self.avefitness = np.mean(self.popFitness)
            self.trace[self.t, 0] = (1 - self.best.fitness) / self.best.fitness
            self.trace[self.t, 1] = (1 - self.avefitness) / self.avefitness
            print("Generation %d: optimal function value is: %f; average function value is %f" % (
                self.t, self.trace[self.t, 0], self.trace[self.t, 1]))
            self.t += 1

        print("Optimal function value is: %f; " %
              self.trace[self.t - 1, 0])
        print "Optimal solution is:"
        print self.best.chrom
        self.printResult()

    def cloneOperation(self, individuals):
        '''
        clone operation for alforithm immune algorithm
        '''
        newpop = []
        sizeInds = len(individuals)
        for i in xrange(0, sizeInds):
            for j in xrange(0, int(self.params[1] * sizeInds)):
                newpop.append(copy.deepcopy(individuals[i]))
        return newpop

    def selectionOperation(self, flag):
        '''
        selection operation for artificial immune algorithm
        '''
        if flag == 1:
            sortedIdx = np.argsort(-self.clonefit)
            for i in xrange(0, int(self.sizepop*self.sizepop*self.params[1]):
            tmpInd = individuals[sortedIdx[i]]
            if tmpInd.fitness > self.population[i].fitness:
                self.population[i] = tmpInd
                self.popFitness[i] = tmpInd.fitness
        else:
            pass
        newpop = []
        sizeInds = len(individuals)
        fitness = np.zeros(sizeInds)
        for i in xrange(0, sizeInds):
            fitness[i] = individuals[i].fitness
        sortedIdx = np.argsort(-fitness)
        for i in xrange(0, sizeInds):
            tmpInd = individuals[sortedIdx[i]]
            if tmpInd.fitness > self.population[i].fitness:
                self.population[i] = tmpInd
                self.popFitness[i] = tmpInd.fitness

    def mutationOperation(self, individuals):
        '''
        mutation operation for artificial immune algorithm
        '''
        newpop = []
        sizeInds = len(individuals)
        for i in xrange(0, sizeInds):
            newpop.append(copy.deepcopy(individuals[i]))
            r = random.random()
            if r < self.params[0]:
                mutatePos = random.randint(0, self.vardim - 1)
                theta = random.random()
                if theta > 0.5:
                    newpop[i].chrom[mutatePos] = newpop[i].chrom[
                        mutatePos] - (newpop[i].chrom[mutatePos] - self.bound[0, mutatePos]) * (1 - random.random() ** (1 - self.t / self.MAXGEN))
                else:
                    newpop[i].chrom[mutatePos] = newpop[i].chrom[
                        mutatePos] + (self.bound[1, mutatePos] - newpop[i].chrom[mutatePos]) * (1 - random.random() ** (1 - self.t / self.MAXGEN))
                for k in xrange(0, self.vardim):
                    if newpop.chrom[mutatePos] < self.bound[0, mutatePos]:
                        newpop.chrom[mutatePos] = self.bound[0, mutatePos]
                    if newpop.chrom[mutatePos] > self.bound[1, mutatePos]:
                        newpop.chrom[mutatePos] = self.bound[1, mutatePos]
                newpop.calculateFitness()
        return newpop

    def printResult(self):
        '''
        plot the result of the artificial immune algorithm
        '''
        x = np.arange(0, self.MAXGEN)
        y1 = self.trace[:, 0]
        y2 = self.trace[:, 1]
        plt.plot(x, y1, 'r', label='optimal value')
        plt.plot(x, y2, 'g', label='average value')
        plt.xlabel("Iteration")
        plt.ylabel("function value")
        plt.title("Artificial immune algorithm for function optimization")
        plt.legend()
        plt.show()
        
        
if __name__ == "__main__":
 
     bound = np.tile([[-600], [600]], 25)
     aia = AIA(100, 25, bound, 100, [0.9, 0.1])
     aia.solve()
```

## 模拟退火算法

模拟退火算法来源于固体退火原理，是一种基于概率的算法，将固体加温至充分高，再让其徐徐冷却，加温时，固体内部[粒子](https://baike.baidu.com/item/粒子/81757)随[温升](https://baike.baidu.com/item/温升/10468133)变为无序状，内能增大，而徐徐冷却时粒子渐趋有序，在每个温度都达到[平衡态](https://baike.baidu.com/item/平衡态/8965512)，最后在[常温](https://baike.baidu.com/item/常温/1144996)时达到基态，内能减为最小。

```python
from matplotlib import pyplot as plt
import numpy as np
 
def coordinate_init(size):
    #产生坐标字典
    coordinate_dict = {}
    coordinate_dict[0] = (0, 0)#起点是（0，0）
    for i in range(1, size + 1):#顺序标号随机坐标
        coordinate_dict[i] = (np.random.uniform(0, size), np.random.uniform(0, size))
    coordinate_dict[size + 1] = (0, 0)#终点是（0,0)
    return coordinate_dict
 
def distance_matrix(coordinate_dict,size):#生成距离矩阵
    d=np.zeros((size+2,size+2))
    for i in range(size+1):
        for j in range(size+1):
            if(i==j):
                continue
            if(d[i][j]!=0):
                continue
            x1 = coordinate_dict[i][0]
            y1 = coordinate_dict[i][1]
            x2 = coordinate_dict[j][0]
            y2 = coordinate_dict[j][1]
            distance=np.sqrt((x1-x2)**2+(y1-y2)**2)
            if(i==0):
                d[i][j]=d[size+1][j]=d[j][i]=d[j][size+1]=distance
            else:
                d[i][j]=d[j][i]=distance
    return d
 
def path_length(d_matrix,path_list,size):#计算路径长度
    length=0
    for i in range(size+1):
        length+=d_matrix[path_list[i]][path_list[i+1]]
    return length
 
def new_path(path_list,size):
    #二交换法
    change_head = np.random.randint(1,size+1)
    change_tail = np.random.randint(1,size+1)
    if(change_head>change_tail):
        change_head,change_tail=change_tail,change_head
    change_list = path_list[change_head:change_tail + 1]
    change_list.reverse()#change_head与change_tail之间的路径反序
    new_path_list = path_list[:change_head] + change_list + path_list[change_tail + 1:]
    return change_head,change_tail,new_path_list
 
def diff_old_new(d_matrix,path_list,new_path_list,head,tail):#计算新旧路径的长度之差
    old_length=d_matrix[path_list[head-1]][path_list[head]]+d_matrix[path_list[tail]][path_list[tail+1]]
    new_length=d_matrix[new_path_list[head-1]][new_path_list[head]]+d_matrix[new_path_list[tail]][new_path_list[tail+1]]
    delta_p=new_length-old_length
    return delta_p
 
 
T_start=2000#起始温度
T_end=1e-20#结束温度
a=0.995#降温速率
Lk=50#内循环次数,马尔科夫链长
size=20
coordinate_dict=coordinate_init(size)
print(coordinate_dict)#打印坐标字典
path_list=list(range(size+2))#初始化路径
d=distance_matrix(coordinate_dict,size)#距离矩阵的生成
best_path=path_length(d,path_list,size)#初始化最好路径长度
print('初始路径:',path_list)
print('初始路径长度:',best_path)
best_path_temp=[]#记录每个温度下最好路径长度
best_path_list=[]#用于记录历史上最好路径
balanced_path_list=path_list#记录每个温度下的平衡路径
balenced_path_temp=[]#记录每个温度下平衡路径(局部最优)的长度
while T_start>T_end:
    for i in range(Lk):
        head, tail, new_path_list = new_path(path_list, size)
        delta_p = diff_old_new(d, path_list, new_path_list, head, tail)
        if delta_p < 0:#接受状态
            balanced_path_list=path_list = new_path_list
            new_len=path_length(d,path_list,size)
            if(new_len<best_path):
                best_path=new_len
                best_path_list=path_list
        elif np.random.random() < np.exp(-delta_p / T_start):#以概率接受状态
            path_list = new_path_list
    path_list=balanced_path_list#继承该温度下的平衡状态（局部最优）
    T_start*=a#退火
    best_path_temp.append(best_path)
    balenced_path_temp.append(path_length(d,balanced_path_list,size))
print('结束温度的局部最优路径:',balanced_path_list)
print('结束温度的局部最优路径长度:',path_length(d,balanced_path_list,size))
print('最好路径:',best_path_list)
print('最好路径长度:',best_path)
x=[]
y=[]
for point in best_path_list:
    x.append(coordinate_dict[point][0])
    y.append(coordinate_dict[point][1])
plt.figure(1)
plt.subplot(311)
plt.plot(balenced_path_temp)#每个温度下平衡路径长度
plt.subplot(312)
plt.plot(best_path_temp)#每个温度下最好路径长度
plt.subplot(313)
plt.scatter(x,y)
plt.plot(x,y)#路径图
plt.grid()
plt.show()
```

## 蚁群算法

[蚁群算法(AG)](https://link.zhihu.com/?target=https%3A//zh.wikipedia.org/zh-hans/%E8%9A%81%E7%BE%A4%E7%AE%97%E6%B3%95)是一种模拟[蚂蚁](https://link.zhihu.com/?target=https%3A//www.baidu.com/s%3Fwd%3D%E8%9A%82%E8%9A%81%26tn%3D24004469_oem_dg%26rsv_dl%3Dgh_pl_sl_csd)觅食行为的模拟优化算法，它是由意大利学者Dorigo M等人于1991年首先提出，并首先使用在解决TSP（旅行商问题）上

第一蚁群优化算法被称为“蚂蚁系统”，它旨在解决推销员问题，其目标是要找到一系列城市的最短遍历路线。总体算法相对简单，要点如下:

- 一组蚂蚁，每只完成一次城市间的遍历。
- 在每个阶段，蚂蚁根据一些规则选择从一个城市移动到另一个：它必须访问每个城市一次;一个越远的城市被选中的机会越少（能见度更低）
- 在两个城市边际的一边形成的信息素越浓烈，这边被选择的概率越大;如果路程短的话，已经完成旅程的蚂蚁会在所有走过的路径上沉积更多信息素
- 每次迭代后，信息素轨迹挥发

```python
# 参数初始化
(ALPHA,BETA,RHO,Q)=(1.0,2.0,0.5,100.0)
# 城市，蚁群
(city_num,ant_num)=(50,50)
# 初始化城市的位置
distance_x = [
    178,272,176,171,650,499,267,703,408,437,491,74,532,
    416,626,42,271,359,163,508,229,576,147,560,35,714,
    757,517,64,314,675,690,391,628,87,240,705,699,258,
    428,614,36,360,482,666,597,209,201,492,294]
distance_y = [
    170,395,198,151,242,556,57,401,305,421,267,105,525,
    381,244,330,395,169,141,380,153,442,528,329,232,48,
    498,265,343,120,165,50,433,63,491,275,348,222,288,
    490,213,524,244,114,104,552,70,425,227,331]
# 城市距离和信息素
distance_graph=[[0.0 for col in range(city_num)] for raw in range(city_num)]
pheromone_graph=[[1.0 for col in range(city_num)] for raw in range(city_num)]

# 选择下一个城市
    def __choice_next_city(self):
        next_city=-1
        # 存储选择下一个城市的概率
        select_city_prob=[0.0 for i in range(city_num)]
        total_prob=0.0
        # 获取下一个城市的概率
        for i in range(city_num):
            if self.open_table_city[i]:
                try:
                    # 计算概率
                    select_city_prob[i]=pow(pheromone_graph[self.current_city][i],ALPHA)*pow(distance_graph[self.current_city][i],BETA)
                    total_prob+=select_city_prob[i]
                except ZeroDivisionError as e:
                    print("分母为0")
                    sys.exit(1)
        # 轮盘选择城市
        if total_prob>0.0:
            # 产生一个随机概率
            temp_prob=random.uniform(0.0,total_prob)
            for i in range(city_num):
                # 轮次递减
                temp_prob-=select_city_prob[i]
                if temp_prob<0.0:
                    next_city=i
                    break
        if (next_city==-1):
            next_city=random.randint(0,city_num-1)
            while self.open_table_city[next_city]==False:
                next_city=random.randint(0,city_num-1)
        # 返回选择的下一个城市号
        return next_city
        
# 更新信息素
    def __update_pheromone_gragh(self):

        # 获取每只蚂蚁在其路径上留下的信息素
        temp_pheromone = [[0.0 for col in range(city_num)] for raw in range(city_num)]
        for ant in self.ants:
            for i in range(1,city_num):
                start, end = ant.path[i-1], ant.path[i]
                # 在路径上的每两个相邻城市间留下信息素，与路径总距离反比
                temp_pheromone[start][end] += Q / ant.total_distance
                temp_pheromone[end][start] = temp_pheromone[start][end]

        # 更新所有城市之间的信息素，旧信息素衰减加上新迭代信息素
        for i in range(city_num):
            for j in range(city_num):
                pheromone_graph[i][j] = pheromone_graph[i][j] * RHO + temp_pheromone[i][j]
```

# 备注

还有一些比较专的算法，调调库就好，就不一一列举了。