import pandas as pd
from sklearn import metrics
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn import svm
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import learning_curve
from sklearn.model_selection import ShuffleSplit
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_score
from sklearn import datasets

data = datasets.load_iris()
X = data['data']
Y = data['target']

# 最简单用法
def Log_():
    clf = LogisticRegression(max_iter=200)
    clf.fit(X, Y)
    preds = clf.predict(X)
    print('准曲率为',metrics.accuracy_score(preds, Y))

# 保存和读取模型
import pickle
with open('mymodel.pickle','wb') as f:
        pickle.dump(clf,f)
with open('mymodel.pickle', 'rb') as f:
    clf2 = pickle.load(f)
clf2.score(X,Y)

# 网格搜索,暴力循环参数，其中cv已经是用了交叉验证
def svm_():
    parameters = {'kernel':['linear', 'rbf'], 'C': range(1,10,1), 'gamma':np.arange(0.1,1.0,0.1)}
    svc = svm.SVC()
    clf = GridSearchCV(svc, parameters, cv=5,scoring='accuracy')
    clf.fit(X, Y)
    preds = clf.predict(X)
    print(clf.best_params_)
    print('模型及其参数', clf.best_estimator_)
    print('准曲率为', metrics.accuracy_score(preds, Y))
    print('混淆矩阵')
    print(metrics.confusion_matrix(Y, preds,labels=[1,2,0]))
    print('混合打分')
    print(metrics.classification_report(Y, preds))


def plot_learning_curve(estimator, title, X, y, axes=None, ylim=None, cv=None,
                        n_jobs=None, train_sizes=np.linspace(.1, 1.0, 5)):
    """
    Generate 3 plots: the test and training learning curve, the training
    samples vs fit times curve, the fit times vs score curve.

    Parameters
    ----------
    estimator : object type that implements the "fit" and "predict" methods
        An object of that type which is cloned for each validation.

    title : string
        Title for the chart.

    X : array-like, shape (n_samples, n_features)
        Training vector, where n_samples is the number of samples and
        n_features is the number of features.

    y : array-like, shape (n_samples) or (n_samples, n_features), optional
        Target relative to X for classification or regression;
        None for unsupervised learning.

    axes : array of 3 axes, optional (default=None)
        Axes to use for plotting the curves.

    ylim : tuple, shape (ymin, ymax), optional
        Defines minimum and maximum yvalues plotted.

    cv : int, cross-validation generator or an iterable, optional
        Determines the cross-validation splitting strategy.
        Possible inputs for cv are:

          - None, to use the default 5-fold cross-validation,
          - integer, to specify the number of folds.
          - :term:`CV splitter`,
          - An iterable yielding (train, test) splits as arrays of indices.

        For integer/None inputs, if ``y`` is binary or multiclass,
        :class:`StratifiedKFold` used. If the estimator is not a classifier
        or if ``y`` is neither binary nor multiclass, :class:`KFold` is used.

        Refer :ref:`User Guide <cross_validation>` for the various
        cross-validators that can be used here.

    n_jobs : int or None, optional (default=None)
        Number of jobs to run in parallel.
        ``None`` means 1 unless in a :obj:`joblib.parallel_backend` context.
        ``-1`` means using all processors. See :term:`Glossary <n_jobs>`
        for more details.

    train_sizes : array-like, shape (n_ticks,), dtype float or int
        Relative or absolute numbers of training examples that will be used to
        generate the learning curve. If the dtype is float, it is regarded as a
        fraction of the maximum size of the training set (that is determined
        by the selected validation method), i.e. it has to be within (0, 1].
        Otherwise it is interpreted as absolute sizes of the training sets.
        Note that for classification the number of samples usually have to
        be big enough to contain at least one sample from each class.
        (default: np.linspace(0.1, 1.0, 5))
    """
    if axes is None:
        _, axes = plt.subplots(1, 3, figsize=(20, 5))

    axes[0].set_title(title)
    if ylim is not None:
        axes[0].set_ylim(*ylim)
    axes[0].set_xlabel("Training examples")
    axes[0].set_ylabel("Score")

    train_sizes, train_scores, test_scores, fit_times, _ = \
        learning_curve(estimator, X, y, cv=cv, n_jobs=n_jobs,
                       train_sizes=train_sizes,
                       return_times=True)
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)
    fit_times_mean = np.mean(fit_times, axis=1)
    fit_times_std = np.std(fit_times, axis=1)

    # Plot learning curve
    axes[0].grid()
    axes[0].fill_between(train_sizes, train_scores_mean - train_scores_std,
                         train_scores_mean + train_scores_std, alpha=0.1,
                         color="r")
    axes[0].fill_between(train_sizes, test_scores_mean - test_scores_std,
                         test_scores_mean + test_scores_std, alpha=0.1,
                         color="g")
    axes[0].plot(train_sizes, train_scores_mean, 'o-', color="r",
                 label="Training score")
    axes[0].plot(train_sizes, test_scores_mean, 'o-', color="g",
                 label="Cross-validation score")
    axes[0].legend(loc="best")

    # Plot n_samples vs fit_times
    axes[1].grid()
    axes[1].plot(train_sizes, fit_times_mean, 'o-')
    axes[1].fill_between(train_sizes, fit_times_mean - fit_times_std,
                         fit_times_mean + fit_times_std, alpha=0.1)
    axes[1].set_xlabel("Training examples")
    axes[1].set_ylabel("fit_times")
    axes[1].set_title("Scalability of the model")

    # Plot fit_time vs score
    axes[2].grid()
    axes[2].plot(fit_times_mean, test_scores_mean, 'o-')
    axes[2].fill_between(fit_times_mean, test_scores_mean - test_scores_std,
                         test_scores_mean + test_scores_std, alpha=0.1)
    axes[2].set_xlabel("fit_times")
    axes[2].set_ylabel("Score")
    axes[2].set_title("Performance of the model")

    return plt
# 画学习曲线
def get_curve():
    fig, axes = plt.subplots(3, 2, figsize=(10, 25))

    x_, y_ = X, Y


    title = "Learning Curves (Naive Bayes)"
    # Cross validation with 100 iterations to get smoother mean test and train
    # score curves, each time with 20% data randomly selected as a validation set.
    cv = ShuffleSplit(n_splits=100, test_size=0.2, random_state=0)

    estimator = GaussianNB()
    plot_learning_curve(estimator, title, x_, y_, axes=axes[:, 0], ylim=(0.7, 1.01),
                        cv=cv, n_jobs=4)

    title = r"Learning Curves (SVM, RBF kernel, $\gamma=0.2$)"
    # SVC is more expensive so we do a lower number of CV iterations:
    cv = ShuffleSplit(n_splits=10, test_size=0.2, random_state=0)
    estimator = svm.SVC(gamma=0.2, kernel='rbf')
    plot_learning_curve(estimator, title, x_, y_, axes=axes[:, 1], ylim=(0.7, 1.01),
                        cv=cv, n_jobs=4)

    plt.show()

#交叉验证
def cross():
    clf = KNeighborsClassifier()
    scores = cross_val_score(clf, X, Y, cv=5, scoring='accuracy')
    print(scores)


if __name__ =="__main__":
    print('------------------------------------------------------------------------------------------------------------')
    Log_()
    print('------------------------------------------------------------------------------------------------------------')
    svm_()
    print('------------------------------------------------------------------------------------------------------------')
    get_curve()
    print('------------------------------------------------------------------------------------------------------------')
    cross()
