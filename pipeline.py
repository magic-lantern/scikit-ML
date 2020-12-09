import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from mpl_toolkits import mplot3d
import seaborn as sns
from sklearn.decomposition import PCA, IncrementalPCA
from sklearn import preprocessing
from sklearn.preprocessing import StandardScaler
import sklearn.cluster as cluster
from sklearn.metrics import adjusted_rand_score, adjusted_mutual_info_score, accuracy_score, confusion_matrix, plot_roc_curve, roc_auc_score, roc_curve, auc
from sklearn.base import clone
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import RFE, RFECV
from sklearn.pipeline import Pipeline
from itertools import combinations
from pyspark.sql import functions as F
from pyspark.sql.functions import max, mean, min, stddev, lit, regexp_replace, col

# set standard random state for repeatability
my_random_state = 42

# from https://github.com/rasbt/python-machine-learning-book-3rd-edition/blob/master/ch03/ch03.ipynb
def plot_decision_regions(X, y, classifier, test_idx=None, resolution=0.02):

    # setup marker generator and color map
    markers = ('s', 'x', 'o', '^', 'v')
    colors = ('red', 'blue', 'lightgreen', 'gray', 'cyan')
    cmap = ListedColormap(colors[:len(np.unique(y))])

    # plot the decision surface
    x1_min, x1_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    x2_min, x2_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx1, xx2 = np.meshgrid(np.arange(x1_min, x1_max, resolution),
                           np.arange(x2_min, x2_max, resolution))
    Z = classifier.predict(np.array([xx1.ravel(), xx2.ravel()]).T)
    Z = Z.reshape(xx1.shape)
    plt.contourf(xx1, xx2, Z, alpha=0.3, cmap=cmap)
    plt.xlim(xx1.min(), xx1.max())
    plt.ylim(xx2.min(), xx2.max())

    for idx, cl in enumerate(np.unique(y)):
        plt.scatter(x=X[y == cl, 0], 
                    y=X[y == cl, 1],
                    alpha=0.8, 
                    c=colors[idx],
                    marker=markers[idx], 
                    label=cl, 
                    edgecolor='black')

    # highlight test examples
    if test_idx:
        # plot all examples
        X_test, y_test = X[test_idx, :], y[test_idx]

        plt.scatter(X_test[:, 0],
                    X_test[:, 1],
                    c='',
                    edgecolor='black',
                    alpha=1.0,
                    linewidth=1,
                    marker='o',
                    s=100, 
                    label='test set')

class SBS():
    def __init__(self, estimator, k_features, scoring=accuracy_score,
                 test_size=0.25, random_state=1):
        self.scoring = scoring
        self.estimator = clone(estimator)
        self.k_features = k_features
        self.test_size = test_size
        self.random_state = random_state

    def fit(self, X, y):
        
        X_train, X_test, y_train, y_test = \
            train_test_split(X, y, test_size=self.test_size,
                             random_state=self.random_state)

        dim = X_train.shape[1]
        self.indices_ = tuple(range(dim))
        self.subsets_ = [self.indices_]
        score = self._calc_score(X_train, y_train, 
                                 X_test, y_test, self.indices_)
        self.scores_ = [score]

        while dim > self.k_features:
            scores = []
            subsets = []

            for p in combinations(self.indices_, r=dim - 1):
                score = self._calc_score(X_train, y_train, 
                                         X_test, y_test, p)
                scores.append(score)
                subsets.append(p)

            best = np.argmax(scores)
            self.indices_ = subsets[best]
            self.subsets_.append(self.indices_)
            dim -= 1

            self.scores_.append(scores[best])
        self.k_score_ = self.scores_[-1]

        return self

    def transform(self, X):
        return X[:, self.indices_]

    def _calc_score(self, X_train, y_train, X_test, y_test, indices):
        self.estimator.fit(X_train[:, indices], y_train)
        y_pred = self.estimator.predict(X_test[:, indices])
        score = self.scoring(y_test, y_pred)
        return score

@transform_pandas(
    Output(rid="ri.foundry.main.dataset.b474df3d-909d-4a81-9e38-515e22b9cff3"),
    inpatient_scaled_w_imputation=Input(rid="ri.foundry.main.dataset.f410db35-59e0-4b82-8fa8-d6dc6a61c9f2"),
    outcomes=Input(rid="ri.foundry.main.dataset.3d9b1654-3923-484f-8db5-6b38b56e290c")
)
def data_and_outcomes(inpatient_scaled_w_imputation, outcomes):
    i = inpatient_scaled_w_imputation
    o = outcomes
    return i.join(o, on=['visit_occurrence_id'], how='inner')

@transform_pandas(
    Output(rid="ri.foundry.main.dataset.6539e1fc-4c2d-47c1-bc55-96268abaa9ea"),
    data_and_outcomes=Input(rid="ri.foundry.main.dataset.b474df3d-909d-4a81-9e38-515e22b9cff3"),
    inpatient_scaled_w_imputation=Input(rid="ri.foundry.main.dataset.f410db35-59e0-4b82-8fa8-d6dc6a61c9f2"),
    outcomes=Input(rid="ri.foundry.main.dataset.3d9b1654-3923-484f-8db5-6b38b56e290c")
)
def lr_rfe(data_and_outcomes, inpatient_scaled_w_imputation, outcomes):
    my_data = data_and_outcomes.select(inpatient_scaled_w_imputation.columns).toPandas()
    my_data = my_data.drop(columns='visit_occurrence_id')
    my_outcomes = data_and_outcomes.select(outcomes.columns).toPandas()
    y = my_outcomes.bad_outcome
    x_train, x_test, y_train, y_test = train_test_split(my_data, y, test_size=0.3, random_state=1, stratify=y)

    lr = LogisticRegression(penalty='l2',
                            C=100.0,
                            random_state=my_random_state,
                            max_iter=10000)
    rfe = RFE(lr, n_features_to_select=40, step=1)

    pipeline = Pipeline(steps=[('s',rfe),('m',lr)])
    pipeline.fit(x_train, y_train)

    # summarize the selection of the attributes
    print(rfe.support_)
    print(rfe.ranking_)
    print(x_test.loc[:, rfe.support_].columns)
    print('coefficients:', pipeline._final_estimator.coef_)

    y_pred = pipeline.predict(x_test)
    confmat = confusion_matrix(y_true=y_test, y_pred=y_pred)
    print('lr with rfe and 40 features')
    print(confmat)

@transform_pandas(
    Output(rid="ri.foundry.main.dataset.32b0e775-ba50-44e2-ae82-5f41ec31a84c"),
    data_and_outcomes=Input(rid="ri.foundry.main.dataset.b474df3d-909d-4a81-9e38-515e22b9cff3"),
    inpatient_scaled_w_imputation=Input(rid="ri.foundry.main.dataset.f410db35-59e0-4b82-8fa8-d6dc6a61c9f2"),
    outcomes=Input(rid="ri.foundry.main.dataset.3d9b1654-3923-484f-8db5-6b38b56e290c")
)
def lr_rfecv(data_and_outcomes, inpatient_scaled_w_imputation, outcomes):
    my_data = data_and_outcomes.select(inpatient_scaled_w_imputation.columns).toPandas()
    my_data = my_data.drop(columns='visit_occurrence_id')
    my_outcomes = data_and_outcomes.select(outcomes.columns).toPandas()
    y = my_outcomes.bad_outcome
    x_train, x_test, y_train, y_test = train_test_split(my_data, y, test_size=0.3, random_state=1, stratify=y)

    # setup model and recursive feature eliminator
    lr = LogisticRegression(penalty='l2',
                            C=100.0,
                            random_state=my_random_state,
                            max_iter=10000)
    rfecv = RFECV(lr, step=1, cv=10)
    pipeline = Pipeline(steps=[('s',rfecv),('m',lr)])
    pipeline.fit(x_train, y_train)

    # summarize the selection of the attributes
    print(rfecv.support_)
    print(rfecv.ranking_)
    print(x_test.loc[:, rfecv.support_].columns)
    print('coefficients:', pipeline._final_estimator.coef_)

    y_pred = pipeline.predict(x_test)
    confmat = confusion_matrix(y_true=y_test, y_pred=y_pred)
    print('lr with rfe 10-fold cv selection of features')
    print(confmat)

    lr_disp = plot_roc_curve(pipeline._final_estimator, x_test.loc[:, rfecv.support_], y_test)
    plt.show()

    # return dataframe with relevant features and associated coefficients
    return pd.DataFrame(data=pipeline._final_estimator.coef_, columns=x_test.loc[:, rfecv.support_].columns)

@transform_pandas(
    Output(rid="ri.foundry.main.dataset.58c8d23e-5558-4347-98c6-e2dc0c7a6ef7"),
    outcomes=Input(rid="ri.foundry.main.dataset.3d9b1654-3923-484f-8db5-6b38b56e290c"),
    pca_rfecv_cols=Input(rid="ri.foundry.main.dataset.c8cf31b6-e5d3-4e91-a06e-d634ec5ce318")
)
def pca_rfecv_bad_outcome(pca_rfecv_cols, outcomes):
    embedding = pca_rfecv_cols.values
    dfo = outcomes.toPandas()

    splt = sns.scatterplot(x = embedding[:, 0],
                            y = embedding[:, 1],
                            hue = dfo.bad_outcome,
                            alpha = 0.6)
    plt.title('PCA w/LR Features UMAP 2D scatter plot')
    plt.show()
    
    return

@transform_pandas(
    Output(rid="ri.foundry.main.dataset.c8cf31b6-e5d3-4e91-a06e-d634ec5ce318"),
    data_and_outcomes=Input(rid="ri.foundry.main.dataset.b474df3d-909d-4a81-9e38-515e22b9cff3"),
    lr_rfecv=Input(rid="ri.foundry.main.dataset.32b0e775-ba50-44e2-ae82-5f41ec31a84c")
)
def pca_rfecv_cols(data_and_outcomes, lr_rfecv):
    arr = data_and_outcomes.select(list(lr_rfecv.columns)).toPandas().values
    
    pca_all = PCA(random_state=42)
    pca_all.fit(arr)
    pca_all_arr = pca_all.transform(arr)

    return pd.DataFrame(pca_all_arr)

@transform_pandas(
    Output(rid="ri.foundry.main.dataset.ca533b97-fde4-4d3f-a987-b2372e7f2894"),
    data_and_outcomes=Input(rid="ri.foundry.main.dataset.b474df3d-909d-4a81-9e38-515e22b9cff3"),
    inpatient_scaled_w_imputation=Input(rid="ri.foundry.main.dataset.f410db35-59e0-4b82-8fa8-d6dc6a61c9f2"),
    outcomes=Input(rid="ri.foundry.main.dataset.3d9b1654-3923-484f-8db5-6b38b56e290c")
)
def sbs_knn(data_and_outcomes, inpatient_scaled_w_imputation, outcomes):
    my_data = data_and_outcomes.select(inpatient_scaled_w_imputation.columns).toPandas()
    my_data = my_data.drop(columns='visit_occurrence_id').values
    my_outcomes = data_and_outcomes.select(outcomes.columns).toPandas()
    y = my_outcomes.bad_outcome.values
    x_train, x_test, y_train, y_test = train_test_split(my_data, y, test_size=0.3, random_state=1, stratify=y)

    knn = KNeighborsClassifier(n_neighbors=5)

    # selecting features
    sbs = SBS(knn, k_features=1)
    # now see how it does
    sbs.fit(x_train, y_train)

    # plotting performance of feature subsets
    k_feat = [len(k) for k in sbs.subsets_]

    plt.plot(k_feat, sbs.scores_, marker='o')
    #plt.ylim([0.7, 1.02])
    plt.ylabel('Accuracy')
    plt.xlabel('Number of features')
    plt.grid()
    plt.tight_layout()
    # plt.savefig('images/04_08.png', dpi=300)
    plt.show()

    #y_pred = sbs.predict(x_test)
    #confmat = confusion_matrix(y_true=y_test, y_pred=y_pred)
    #print('knn 5 with sequential backward selection')
    #print(confmat)

#     knn = KNeighborsClassifier(n_neighbors=10)
#     # selecting features
#     sbs = SBS(knn, k_features=1)
#     sbs.fit(x_train, y_train)
#     y_pred = sbs.predict(x_test)
#     confmat = confusion_matrix(y_true=y_test, y_pred=y_pred)
#     print('knn 10 with sequential backward selection')
#     print(confmat)

#     # plotting performance of feature subsets
#     k_feat = [len(k) for k in sbs.subsets_]

#     plt.plot(k_feat, sbs.scores_, marker='o')
#     plt.ylim([0.7, 1.02])
#     plt.ylabel('Accuracy')
#     plt.xlabel('Number of features')
#     plt.grid()
#     plt.tight_layout()
#     # plt.savefig('images/04_08.png', dpi=300)
#     plt.show()

#     plot_decision_regions(my_data.values, y.values, classifier=lr)
#     plt.xlabel('petal length [standardized]')
#     plt.ylabel('petal width [standardized]')
#     plt.legend(loc='upper left')
#     plt.tight_layout()
#     plt.show()

@transform_pandas(
    Output(rid="ri.foundry.main.dataset.e0fd8f16-a131-4276-84c7-acc20e7f1829"),
    data_and_outcomes=Input(rid="ri.foundry.main.dataset.b474df3d-909d-4a81-9e38-515e22b9cff3"),
    inpatient_scaled_w_imputation=Input(rid="ri.foundry.main.dataset.f410db35-59e0-4b82-8fa8-d6dc6a61c9f2"),
    outcomes=Input(rid="ri.foundry.main.dataset.3d9b1654-3923-484f-8db5-6b38b56e290c")
)
def various_lr(data_and_outcomes, inpatient_scaled_w_imputation, outcomes):

    my_data = data_and_outcomes.select(inpatient_scaled_w_imputation.columns).toPandas()
    my_data = my_data.drop(columns='visit_occurrence_id')
    my_outcomes = data_and_outcomes.select(outcomes.columns).toPandas()
    y = my_outcomes.bad_outcome
    x_train, x_test, y_train, y_test = train_test_split(my_data, y, test_size=0.3, random_state=1, stratify=y)

    lr = LogisticRegression(penalty='none',
                            random_state=my_random_state,
                            max_iter=10000)
    lr.fit(x_train, y_train)

    y_pred = lr.predict(x_test)
    confmat = confusion_matrix(y_true=y_test, y_pred=y_pred)
    print('lr with no penalty')
    print(confmat)

    lr = LogisticRegression(penalty='l1',
                            solver='saga',
                            C=100.0,
                            random_state=my_random_state,
                            max_iter=10000)
    lr.fit(x_train, y_train)

    y_pred = lr.predict(x_test)
    confmat = confusion_matrix(y_true=y_test, y_pred=y_pred)
    print('lr with saga solver and l1 penalty')
    print(confmat)

    lr = LogisticRegression(penalty='l2',
                            C=100.0,
                            random_state=my_random_state,
                            max_iter=10000)
    lr.fit(x_train, y_train)

    y_pred = lr.predict(x_test)
    confmat = confusion_matrix(y_true=y_test, y_pred=y_pred)
    print('lr with l2 penalty')
    print(confmat)

    lr = LogisticRegression(penalty='elasticnet',
                            solver='saga',
                            l1_ratio=0.0,
                            C=100.0,
                            random_state=my_random_state,
                            max_iter=10000)
    lr.fit(x_train, y_train)

    y_pred = lr.predict(x_test)
    confmat = confusion_matrix(y_true=y_test, y_pred=y_pred)
    print('lr with saga solver and elasticnet penalty l1')
    print(confmat)

    lr = LogisticRegression(penalty='elasticnet',
                            solver='saga',
                            l1_ratio=0.5,
                            C=100.0,
                            random_state=my_random_state,
                            max_iter=10000)
    lr.fit(x_train, y_train)

    y_pred = lr.predict(x_test)
    confmat = confusion_matrix(y_true=y_test, y_pred=y_pred)
    print('lr with saga solver and elasticnet penalty half l1 and l2')
    print(confmat)

    lr = LogisticRegression(penalty='elasticnet',
                            solver='saga',
                            l1_ratio=1.0,
                            C=100.0,
                            random_state=my_random_state,
                            max_iter=10000)
    lr.fit(x_train, y_train)

    y_pred = lr.predict(x_test)
    confmat = confusion_matrix(y_true=y_test, y_pred=y_pred)
    print('lr with saga solver and elasticnet penalty l2')
    print(confmat)

    # need to get this to work for visualization of results
    #plot_decision_regions(my_data.values, y.values, classifier=lr)
    #plt.xlabel('petal length [standardized]')
    #plt.ylabel('petal width [standardized]')
    #plt.legend(loc='upper left')
    #plt.tight_layout()
    #plt.show()

