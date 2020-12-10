import numpy as np
import pandas as pd
import umap
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
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.feature_selection import RFE, RFECV
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
from itertools import combinations
from pyspark.sql import functions as F
from pyspark.sql.functions import max, mean, min, stddev, lit, regexp_replace, col

import timeit

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

import numbers
from sklearn.feature_selection import SelectorMixin
from sklearn.base import BaseEstimator, MetaEstimatorMixin, clone
from sklearn.utils.validation import check_is_fitted
from sklearn.model_selection import cross_val_score

# from https://github.com/scikit-learn/scikit-learn/blob/4773f3e39/sklearn/feature_selection/_sequential.py
class SequentialFeatureSelector(SelectorMixin, MetaEstimatorMixin,
                                BaseEstimator):
    """Transformer that performs Sequential Feature Selection.
    This Sequential Feature Selector adds (forward selection) or
    removes (backward selection) features to form a feature subset in a
    greedy fashion. At each stage, this estimator chooses the best feature to
    add or remove based on the cross-validation score of an estimator.
    Read more in the :ref:`User Guide <sequential_feature_selection>`.
    .. versionadded:: 0.24
    Parameters
    ----------
    estimator : estimator instance
        An unfitted estimator.
    n_features_to_select : int or float, default=None
        The number of features to select. If `None`, half of the features are
        selected. If integer, the parameter is the absolute number of features
        to select. If float between 0 and 1, it is the fraction of features to
        select.
    direction: {'forward', 'backward'}, default='forward'
        Whether to perform forward selection or backward selection.
    scoring : str, callable, list/tuple or dict, default=None
        A single str (see :ref:`scoring_parameter`) or a callable
        (see :ref:`scoring`) to evaluate the predictions on the test set.
        NOTE that when using custom scorers, each scorer should return a single
        value. Metric functions returning a list/array of values can be wrapped
        into multiple scorers that return one value each.
        If None, the estimator's score method is used.
    cv : int, cross-validation generator or an iterable, default=None
        Determines the cross-validation splitting strategy.
        Possible inputs for cv are:
        - None, to use the default 5-fold cross validation,
        - integer, to specify the number of folds in a `(Stratified)KFold`,
        - :term:`CV splitter`,
        - An iterable yielding (train, test) splits as arrays of indices.
        For integer/None inputs, if the estimator is a classifier and ``y`` is
        either binary or multiclass, :class:`StratifiedKFold` is used. In all
        other cases, :class:`KFold` is used.
        Refer :ref:`User Guide <cross_validation>` for the various
        cross-validation strategies that can be used here.
    n_jobs : int, default=None
        Number of jobs to run in parallel. When evaluating a new feature to
        add or remove, the cross-validation procedure is parallel over the
        folds.
        ``None`` means 1 unless in a :obj:`joblib.parallel_backend` context.
        ``-1`` means using all processors. See :term:`Glossary <n_jobs>`
        for more details.
    Attributes
    ----------
    n_features_to_select_ : int
        The number of features that were selected.
    support_ : ndarray of shape (n_features,), dtype=bool
        The mask of selected features.
    See Also
    --------
    RFE : Recursive feature elimination based on importance weights.
    RFECV : Recursive feature elimination based on importance weights, with
        automatic selection of the number of features.
    SelectFromModel : Feature selection based on thresholds of importance
        weights.
    Examples
    --------
    >>> from sklearn.feature_selection import SequentialFeatureSelector
    >>> from sklearn.neighbors import KNeighborsClassifier
    >>> from sklearn.datasets import load_iris
    >>> X, y = load_iris(return_X_y=True)
    >>> knn = KNeighborsClassifier(n_neighbors=3)
    >>> sfs = SequentialFeatureSelector(knn, n_features_to_select=3)
    >>> sfs.fit(X, y)
    SequentialFeatureSelector(estimator=KNeighborsClassifier(n_neighbors=3),
                              n_features_to_select=3)
    >>> sfs.get_support()
    array([ True, False,  True,  True])
    >>> sfs.transform(X).shape
    (150, 3)
    """
    def __init__(self, estimator, *, n_features_to_select=None,
                 direction='forward', scoring=None, cv=5, n_jobs=None):

        self.estimator = estimator
        self.n_features_to_select = n_features_to_select
        self.direction = direction
        self.scoring = scoring
        self.cv = cv
        self.n_jobs = n_jobs

    def fit(self, X, y):
        """Learn the features to select.
        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training vectors.
        y : array-like of shape (n_samples,)
            Target values.
        Returns
        -------
        self : object
        """
        tags = self._get_tags()
        X, y = self._validate_data(
            X, y, accept_sparse="csc",
            ensure_min_features=2,
            force_all_finite=not tags.get("allow_nan", True),
            multi_output=True
        )
        n_features = X.shape[1]

        error_msg = ("n_features_to_select must be either None, an "
                     "integer in [1, n_features - 1] "
                     "representing the absolute "
                     "number of features, or a float in (0, 1] "
                     "representing a percentage of features to "
                     f"select. Got {self.n_features_to_select}")
        if self.n_features_to_select is None:
            self.n_features_to_select_ = n_features // 2
        elif isinstance(self.n_features_to_select, numbers.Integral):
            if not 0 < self.n_features_to_select < n_features:
                raise ValueError(error_msg)
            self.n_features_to_select_ = self.n_features_to_select
        elif isinstance(self.n_features_to_select, numbers.Real):
            if not 0 < self.n_features_to_select <= 1:
                raise ValueError(error_msg)
            self.n_features_to_select_ = int(n_features *
                                             self.n_features_to_select)
        else:
            raise ValueError(error_msg)

        if self.direction not in ('forward', 'backward'):
            raise ValueError(
                "direction must be either 'forward' or 'backward'. "
                f"Got {self.direction}."
            )

        cloned_estimator = clone(self.estimator)

        # the current mask corresponds to the set of features:
        # - that we have already *selected* if we do forward selection
        # - that we have already *excluded* if we do backward selection
        current_mask = np.zeros(shape=n_features, dtype=bool)
        n_iterations = (
            self.n_features_to_select_ if self.direction == 'forward'
            else n_features - self.n_features_to_select_
        )
        for _ in range(n_iterations):
            new_feature_idx = self._get_best_new_feature(cloned_estimator, X,
                                                         y, current_mask)
            current_mask[new_feature_idx] = True

        if self.direction == 'backward':
            current_mask = ~current_mask
        self.support_ = current_mask

        return self

    def _get_best_new_feature(self, estimator, X, y, current_mask):
        # Return the best new feature to add to the current_mask, i.e. return
        # the best new feature to add (resp. remove) when doing forward
        # selection (resp. backward selection)
        candidate_feature_indices = np.flatnonzero(~current_mask)
        scores = {}
        for feature_idx in candidate_feature_indices:
            candidate_mask = current_mask.copy()
            candidate_mask[feature_idx] = True
            if self.direction == 'backward':
                candidate_mask = ~candidate_mask
            X_new = X[:, candidate_mask]
            scores[feature_idx] = cross_val_score(
                estimator, X_new, y, cv=self.cv, scoring=self.scoring,
                n_jobs=self.n_jobs).mean()
        return max(scores, key=lambda feature_idx: scores[feature_idx])

    def _get_support_mask(self):
        check_is_fitted(self)
        return self.support_

    def _more_tags(self):
        estimator_tags = self.estimator._get_tags()
        return {
            'allow_nan': estimator_tags.get('allow_nan', True),
            'requires_y': True,
        }

@transform_pandas(
    Output(rid="ri.foundry.main.dataset.32069249-a675-4faf-9d3c-a68ff0670c07"),
    inpatient_encoded_w_imputation=Input(rid="ri.foundry.main.dataset.d3578a81-014a-49a6-9887-53d296155bdd"),
    outcomes=Input(rid="ri.foundry.main.dataset.3d9b1654-3923-484f-8db5-6b38b56e290c")
)
def data_encoded_and_outcomes(inpatient_encoded_w_imputation, outcomes):
    i = inpatient_encoded_w_imputation
    o = outcomes
    return i.join(o, on=['visit_occurrence_id'], how='inner')

@transform_pandas(
    Output(rid="ri.foundry.main.dataset.b474df3d-909d-4a81-9e38-515e22b9cff3"),
    inpatient_scaled_w_imputation=Input(rid="ri.foundry.main.dataset.f410db35-59e0-4b82-8fa8-d6dc6a61c9f2"),
    outcomes=Input(rid="ri.foundry.main.dataset.3d9b1654-3923-484f-8db5-6b38b56e290c")
)
def data_scaled_and_outcomes(inpatient_scaled_w_imputation, outcomes):
    i = inpatient_scaled_w_imputation
    o = outcomes
    return i.join(o, on=['visit_occurrence_id'], how='inner')

@transform_pandas(
    Output(rid="ri.foundry.main.dataset.6539e1fc-4c2d-47c1-bc55-96268abaa9ea"),
    data_scaled_and_outcomes=Input(rid="ri.foundry.main.dataset.b474df3d-909d-4a81-9e38-515e22b9cff3"),
    inpatient_scaled_w_imputation=Input(rid="ri.foundry.main.dataset.f410db35-59e0-4b82-8fa8-d6dc6a61c9f2"),
    outcomes=Input(rid="ri.foundry.main.dataset.3d9b1654-3923-484f-8db5-6b38b56e290c")
)
def lr_rfe(data_scaled_and_outcomes, inpatient_scaled_w_imputation, outcomes):
    data_and_outcomes = data_scaled_and_outcomes
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
    data_scaled_and_outcomes=Input(rid="ri.foundry.main.dataset.b474df3d-909d-4a81-9e38-515e22b9cff3"),
    inpatient_scaled_w_imputation=Input(rid="ri.foundry.main.dataset.f410db35-59e0-4b82-8fa8-d6dc6a61c9f2"),
    outcomes=Input(rid="ri.foundry.main.dataset.3d9b1654-3923-484f-8db5-6b38b56e290c")
)
def lr_rfecv(data_scaled_and_outcomes, inpatient_scaled_w_imputation, outcomes):
    data_and_outcomes = data_scaled_and_outcomes
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
    pca_rfecv_cols_umap_embedding=Input(rid="ri.foundry.main.dataset.438c95e7-3842-40a2-a718-4e9826193dd4")
)
def pca_rfecv_bad_outcome( outcomes, pca_rfecv_cols_umap_embedding):
    embedding = pca_rfecv_cols_umap_embedding.values
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
    data_scaled_and_outcomes=Input(rid="ri.foundry.main.dataset.b474df3d-909d-4a81-9e38-515e22b9cff3"),
    lr_rfecv=Input(rid="ri.foundry.main.dataset.32b0e775-ba50-44e2-ae82-5f41ec31a84c")
)
def pca_rfecv_cols(data_scaled_and_outcomes, lr_rfecv):
    data_and_outcomes = data_scaled_and_outcomes
    arr = data_and_outcomes.select(list(lr_rfecv.columns)).toPandas().values
    
    pca_all = PCA(random_state=42)
    pca_all.fit(arr)
    pca_all_arr = pca_all.transform(arr)

    return pd.DataFrame(pca_all_arr)

@transform_pandas(
    Output(rid="ri.foundry.main.dataset.438c95e7-3842-40a2-a718-4e9826193dd4"),
    pca_rfecv_cols=Input(rid="ri.foundry.main.dataset.c8cf31b6-e5d3-4e91-a06e-d634ec5ce318")
)
def pca_rfecv_cols_umap_embedding(pca_rfecv_cols):
    scaled_arr = pca_rfecv_cols

    reducer = umap.UMAP(random_state=42, n_neighbors=200, local_connectivity=15)
    reducer.fit(scaled_arr)
    embedding = reducer.transform(scaled_arr)
    return pd.DataFrame(embedding)

@transform_pandas(
    Output(rid="ri.foundry.main.dataset.db9fda68-0d99-4cd5-8bc1-d2b01127a80d"),
    outcomes=Input(rid="ri.foundry.main.dataset.3d9b1654-3923-484f-8db5-6b38b56e290c"),
    pca_rfecv_cols_umap_embedding=Input(rid="ri.foundry.main.dataset.438c95e7-3842-40a2-a718-4e9826193dd4")
)
def pca_rfecv_severity( outcomes, pca_rfecv_cols_umap_embedding):
    embedding = pca_rfecv_cols_umap_embedding.values
    dfo = outcomes.toPandas()
    dfo['severity_type'] = dfo.severity_type.astype('category')

    splt = sns.scatterplot(x = embedding[:, 0],
                            y = embedding[:, 1],
                            hue = dfo.severity_type,
                            alpha = 0.6)
    plt.title('PCA UMAP 2D scatter plot')
    plt.show()
    
    return

@transform_pandas(
    Output(rid="ri.foundry.main.dataset.a7f3c4ad-c91b-4a90-9963-d466eb8f0e9d"),
    outcomes=Input(rid="ri.foundry.main.dataset.3d9b1654-3923-484f-8db5-6b38b56e290c"),
    pca_rfecv_cols_umap_embedding=Input(rid="ri.foundry.main.dataset.438c95e7-3842-40a2-a718-4e9826193dd4")
)
def pca_rfecv_site( outcomes, pca_rfecv_cols_umap_embedding):
    embedding = pca_rfecv_cols_umap_embedding.values
    dfo = outcomes.toPandas()
    dfo['data_partner_id'] = dfo.data_partner_id.astype('category')

    #fig = plt.figure(figsize = (12, 8))
    #ax = plt.axes()
    splt = sns.scatterplot(x = embedding[:, 0],
                            y = embedding[:, 1],
                            hue = dfo.data_partner_id,
                            alpha = 0.6,
                            legend = True)
    plt.legend(bbox_to_anchor=(1, 1), loc="upper left")
    plt.title('PCA UMAP 2D scatter plot')
    plt.show()
    
    return

@transform_pandas(
    Output(rid="ri.foundry.main.dataset.d152ad50-dd8b-4ea2-a0fd-36d3d7ef448d"),
    data_encoded_and_outcomes=Input(rid="ri.foundry.main.dataset.32069249-a675-4faf-9d3c-a68ff0670c07"),
    inpatient_encoded_w_imputation=Input(rid="ri.foundry.main.dataset.d3578a81-014a-49a6-9887-53d296155bdd"),
    outcomes=Input(rid="ri.foundry.main.dataset.3d9b1654-3923-484f-8db5-6b38b56e290c")
)
def rf_best_feat( outcomes, data_encoded_and_outcomes, inpatient_encoded_w_imputation):
    data_and_outcomes = data_encoded_and_outcomes
    my_data = data_and_outcomes.select(inpatient_encoded_w_imputation.columns).toPandas()
    my_data = my_data.drop(columns='visit_occurrence_id')
    my_outcomes = data_and_outcomes.select(outcomes.columns).toPandas()
    y = my_outcomes.bad_outcome
    x_train, x_test, y_train, y_test = train_test_split(my_data, y, test_size=0.3, random_state=1, stratify=y)

    #########################
    rf = RandomForestClassifier(n_estimators=500,
                                random_state=my_random_state,
                                criterion='gini')
    rf.fit(x_train, y_train)

    # summarize the selection of the attributes
    importances = rf.feature_importances_
    indices = np.argsort(importances)[::-1]

    for f in range(x_train.shape[1]):
        print("%2d) %-*s %f" % (f + 1, 30, 
                                my_data.columns[indices[f]], 
                                importances[indices[f]]))

    y_pred = rf.predict(x_test)
    confmat = confusion_matrix(y_true=y_test, y_pred=y_pred)
    print('rf w 500 estimators w/gini')
    print(confmat)
    y_pred = rf.predict_proba(x_test)[:, 1]
    print('ROC_AUC_SCORE: ', roc_auc_score(y_true=y_test, y_score=y_pred))
    #########################

    rf = RandomForestClassifier(n_estimators=250,
                                random_state=my_random_state,
                                criterion='gini')
    rf.fit(x_train, y_train)

    # summarize the selection of the attributes
    importances = rf.feature_importances_
    indices = np.argsort(importances)[::-1]

    for f in range(x_train.shape[1]):
        print("%2d) %-*s %f" % (f + 1, 30, 
                                my_data.columns[indices[f]], 
                                importances[indices[f]]))

    y_pred = rf.predict(x_test)
    confmat = confusion_matrix(y_true=y_test, y_pred=y_pred)
    print('rf w 250 estimators w/gini')
    print(confmat)
    y_pred = rf.predict_proba(x_test)[:, 1]
    print('ROC_AUC_SCORE: ', roc_auc_score(y_true=y_test, y_score=y_pred))
    #########################

    rf = RandomForestClassifier(n_estimators=100,
                                random_state=my_random_state,
                                criterion='gini')
    rf.fit(x_train, y_train)

    # summarize the selection of the attributes
    importances = rf.feature_importances_
    indices = np.argsort(importances)[::-1]

    for f in range(x_train.shape[1]):
        print("%2d) %-*s %f" % (f + 1, 30, 
                                my_data.columns[indices[f]], 
                                importances[indices[f]]))

    y_pred = rf.predict(x_test)
    confmat = confusion_matrix(y_true=y_test, y_pred=y_pred)
    print('rf w 100 estimators w/gini')
    print(confmat)
    y_pred = rf.predict_proba(x_test)[:, 1]
    print('ROC_AUC_SCORE: ', roc_auc_score(y_true=y_test, y_score=y_pred))
    #########################

    #########################
    rf = RandomForestClassifier(n_estimators=500,
                                random_state=my_random_state,
                                criterion='entropy')
    rf.fit(x_train, y_train)

    # summarize the selection of the attributes
    importances = rf.feature_importances_
    indices = np.argsort(importances)[::-1]

    for f in range(x_train.shape[1]):
        print("%2d) %-*s %f" % (f + 1, 30, 
                                my_data.columns[indices[f]], 
                                importances[indices[f]]))

    y_pred = rf.predict(x_test)
    confmat = confusion_matrix(y_true=y_test, y_pred=y_pred)
    print('rf w 500 estimators w/entropy')
    print(confmat)
    y_pred = rf.predict_proba(x_test)[:, 1]
    print('ROC_AUC_SCORE: ', roc_auc_score(y_true=y_test, y_score=y_pred))
    #########################

    rf = RandomForestClassifier(n_estimators=250,
                                random_state=my_random_state,
                                criterion='entropy')
    rf.fit(x_train, y_train)

    # summarize the selection of the attributes
    importances = rf.feature_importances_
    indices = np.argsort(importances)[::-1]

    for f in range(x_train.shape[1]):
        print("%2d) %-*s %f" % (f + 1, 30, 
                                my_data.columns[indices[f]], 
                                importances[indices[f]]))

    y_pred = rf.predict(x_test)
    confmat = confusion_matrix(y_true=y_test, y_pred=y_pred)
    print('rf w 250 estimators w/entropy')
    print(confmat)
    y_pred = rf.predict_proba(x_test)[:, 1]
    print('ROC_AUC_SCORE: ', roc_auc_score(y_true=y_test, y_score=y_pred))
    #########################

    rf = RandomForestClassifier(n_estimators=100,
                                random_state=my_random_state,
                                criterion='entropy')
    rf.fit(x_train, y_train)

    # summarize the selection of the attributes
    importances = rf.feature_importances_
    indices = np.argsort(importances)[::-1]

    for f in range(x_train.shape[1]):
        print("%2d) %-*s %f" % (f + 1, 30, 
                                my_data.columns[indices[f]], 
                                importances[indices[f]]))

    y_pred = rf.predict(x_test)
    confmat = confusion_matrix(y_true=y_test, y_pred=y_pred)
    print('rf w 100 estimators w/entropy')
    print(confmat)
    y_pred = rf.predict_proba(x_test)[:, 1]
    print('ROC_AUC_SCORE: ', roc_auc_score(y_true=y_test, y_score=y_pred))
    #########################

    fig, (ax1, ax2) = plt.subplots(2, figsize=(7,10))
    fig.tight_layout(h_pad=4)

    rf_disp = plot_roc_curve(rf, x_test, y_test, ax=ax1)

    ax2.set_title('Feature Importance')
    ax2.bar(range(x_train.shape[1]), 
            importances[indices],
            align='center')

    #plt.sca(ax2)
    #ax2.set_xticklabels(my_data.columns[indices])
    #plt.xticks(range(x_train.shape[1]), my_data.columns[indices], rotation=90)
    ax2.set_xticks(range(x_train.shape[1]))
    ax2.set_xticklabels(list(my_data.columns[indices]), rotation=-45, ha='left', fontsize=6)
    ax2.set_xlim([-1, x_train.shape[1]])

    plt.subplots_adjust(bottom=0.2)

    plt.show()

@transform_pandas(
    Output(rid="ri.foundry.main.dataset.13b585c9-f065-4065-bffe-dd4abaafcc56"),
    data_encoded_and_outcomes=Input(rid="ri.foundry.main.dataset.32069249-a675-4faf-9d3c-a68ff0670c07"),
    inpatient_encoded_w_imputation=Input(rid="ri.foundry.main.dataset.d3578a81-014a-49a6-9887-53d296155bdd"),
    outcomes=Input(rid="ri.foundry.main.dataset.3d9b1654-3923-484f-8db5-6b38b56e290c")
)
def rf_gs( outcomes, data_encoded_and_outcomes, inpatient_encoded_w_imputation):
    data_and_outcomes = data_encoded_and_outcomes
    my_data = data_and_outcomes.select(inpatient_encoded_w_imputation.columns).toPandas()
    my_data = my_data.drop(columns='visit_occurrence_id')
    my_outcomes = data_and_outcomes.select(outcomes.columns).toPandas()
    y = my_outcomes.bad_outcome
    x_train, x_test, y_train, y_test = train_test_split(my_data, y, test_size=0.3, random_state=1, stratify=y)

    parameters = {
        'n_estimators':[100,250,500,750],
        'criterion': ['gini', 'entropy'],
        'min_samples_split': [2, 5, 10, 20],
        'max_features' : ['sqrt', 'log2']
    }

    rf = RandomForestClassifier(random_state=my_random_state)
    gd = GridSearchCV(estimator=rf, param_grid=parameters, cv=5)
    gd.fit(x_train, y_train)
    print(gd.best_params_)
    

    # rf.fit(x_train, y_train)

    # # summarize the selection of the attributes
    # importances = rf.feature_importances_
    # indices = np.argsort(importances)[::-1]

    # for f in range(x_train.shape[1]):
    #     print("%2d) %-*s %f" % (f + 1, 30, 
    #                             my_data.columns[indices[f]], 
    #                             importances[indices[f]]))

    # y_pred = rf.predict(x_test)
    # confmat = confusion_matrix(y_true=y_test, y_pred=y_pred)
    # print('rf w 500 estimators w/gini')
    # print(confmat)

    # y_pred = rf.predict_proba(x_test)[:, 1]
    # fpr, tpr, thresholds = roc_curve(y_true=y_test, y_score=y_pred)
    # print('AUC:', auc(x=fpr, y=tpr))
    # print('ROC_AUC_SCORE: ', roc_auc_score(y_true=y_test, y_score=y_pred))

    # fig, (ax1, ax2) = plt.subplots(2, figsize=(7,10))
    # fig.tight_layout(h_pad=4)

    # rf_disp = plot_roc_curve(rf, x_test, y_test, ax=ax1)

    # ax2.set_title('Feature Importance')
    # ax2.bar(range(x_train.shape[1]), 
    #         importances[indices],
    #         align='center')

    # #plt.sca(ax2)
    # #ax2.set_xticklabels(my_data.columns[indices])
    # #plt.xticks(range(x_train.shape[1]), my_data.columns[indices], rotation=90)
    # ax2.set_xticks(range(x_train.shape[1]))
    # ax2.set_xticklabels(list(my_data.columns[indices]), rotation=-45, ha='left', fontsize=6)
    # ax2.set_xlim([-1, x_train.shape[1]])

    # plt.subplots_adjust(bottom=0.2)

    # plt.show()

@transform_pandas(
    Output(rid="ri.vector.main.execute.e41b1b43-b0c5-40c3-aedc-1807d1433e1b"),
    data_encoded_and_outcomes=Input(rid="ri.foundry.main.dataset.32069249-a675-4faf-9d3c-a68ff0670c07"),
    inpatient_encoded_w_imputation=Input(rid="ri.foundry.main.dataset.d3578a81-014a-49a6-9887-53d296155bdd"),
    outcomes=Input(rid="ri.foundry.main.dataset.3d9b1654-3923-484f-8db5-6b38b56e290c")
)
def rf_gs_1( outcomes, data_encoded_and_outcomes, inpatient_encoded_w_imputation):
    start = timeit.default_timer()
    data_and_outcomes = data_encoded_and_outcomes
    my_data = data_and_outcomes.select(inpatient_encoded_w_imputation.columns).toPandas()
    my_data = my_data.drop(columns='visit_occurrence_id')
    my_outcomes = data_and_outcomes.select(outcomes.columns).toPandas()
    y = my_outcomes.bad_outcome
    x_train, x_test, y_train, y_test = train_test_split(my_data, y, test_size=0.3, random_state=1, stratify=y)

    parameters = {
        'n_estimators':[100,250,500,750,1000],
        'criterion': ['gini', 'entropy'],
        'min_samples_split': [2, 5, 10, 20],
        'max_features' : ['sqrt', 'log2']
    }

    rf = RandomForestClassifier(random_state=my_random_state)
    gd = GridSearchCV(estimator=rf, param_grid=parameters, cv=5, n_jobs=10)
    # with n_jobs=5
    # {'criterion': 'gini', 'max_features': 'sqrt', 'min_samples_split': 5, 'n_estimators': 750}
    # Time:  1187.8493531020358

    gd.fit(x_train, y_train)
    print(gd.best_params_)
    

    # rf.fit(x_train, y_train)

    # # summarize the selection of the attributes
    # importances = rf.feature_importances_
    # indices = np.argsort(importances)[::-1]

    # for f in range(x_train.shape[1]):
    #     print("%2d) %-*s %f" % (f + 1, 30, 
    #                             my_data.columns[indices[f]], 
    #                             importances[indices[f]]))

    # y_pred = rf.predict(x_test)
    # confmat = confusion_matrix(y_true=y_test, y_pred=y_pred)
    # print('rf w 500 estimators w/gini')
    # print(confmat)

    # y_pred = rf.predict_proba(x_test)[:, 1]
    # fpr, tpr, thresholds = roc_curve(y_true=y_test, y_score=y_pred)
    # print('AUC:', auc(x=fpr, y=tpr))
    # print('ROC_AUC_SCORE: ', roc_auc_score(y_true=y_test, y_score=y_pred))

    # fig, (ax1, ax2) = plt.subplots(2, figsize=(7,10))
    # fig.tight_layout(h_pad=4)

    # rf_disp = plot_roc_curve(rf, x_test, y_test, ax=ax1)

    # ax2.set_title('Feature Importance')
    # ax2.bar(range(x_train.shape[1]), 
    #         importances[indices],
    #         align='center')

    # #plt.sca(ax2)
    # #ax2.set_xticklabels(my_data.columns[indices])
    # #plt.xticks(range(x_train.shape[1]), my_data.columns[indices], rotation=90)
    # ax2.set_xticks(range(x_train.shape[1]))
    # ax2.set_xticklabels(list(my_data.columns[indices]), rotation=-45, ha='left', fontsize=6)
    # ax2.set_xlim([-1, x_train.shape[1]])

    # plt.subplots_adjust(bottom=0.2)

    # plt.show()

    stop = timeit.default_timer()
    print('Time: ', stop - start)  

@transform_pandas(
    Output(rid="ri.foundry.main.dataset.ca533b97-fde4-4d3f-a987-b2372e7f2894"),
    data_scaled_and_outcomes=Input(rid="ri.foundry.main.dataset.b474df3d-909d-4a81-9e38-515e22b9cff3"),
    inpatient_scaled_w_imputation=Input(rid="ri.foundry.main.dataset.f410db35-59e0-4b82-8fa8-d6dc6a61c9f2"),
    outcomes=Input(rid="ri.foundry.main.dataset.3d9b1654-3923-484f-8db5-6b38b56e290c")
)
def sbs_knn(data_scaled_and_outcomes, inpatient_scaled_w_imputation, outcomes):
    data_and_outcomes = data_scaled_and_outcomes
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
    Output(rid="ri.foundry.main.dataset.2fceafbf-2355-4cfc-b70b-843b185f2a58"),
    data_scaled_and_outcomes=Input(rid="ri.foundry.main.dataset.b474df3d-909d-4a81-9e38-515e22b9cff3"),
    inpatient_scaled_w_imputation=Input(rid="ri.foundry.main.dataset.f410db35-59e0-4b82-8fa8-d6dc6a61c9f2"),
    outcomes=Input(rid="ri.foundry.main.dataset.3d9b1654-3923-484f-8db5-6b38b56e290c")
)
def svm_gs(data_scaled_and_outcomes, outcomes, inpatient_scaled_w_imputation):
    start = timeit.default_timer()

    data_and_outcomes = data_scaled_and_outcomes
    my_data = data_and_outcomes.select(inpatient_scaled_w_imputation.columns).toPandas()
    my_data = my_data.drop(columns='visit_occurrence_id')
    my_outcomes = data_and_outcomes.select(outcomes.columns).toPandas()
    y = my_outcomes.bad_outcome
    x_train, x_test, y_train, y_test = train_test_split(my_data, y, test_size=0.3, random_state=1, stratify=y)

    param_range = [0.0001, 0.001, 0.01, 0.1, 1.0, 10.0, 100.0, 1000.0]
    parameters = {
        'kernel':['linear', 'poly', 'rbf', 'sigmoid'],
        #'gamma': ['scale', 'auto', 0.1, 0.2, 1.0, 10.0],
        #'C': param_range
    }

    # run time with default env and cache_size 1600 - 376 sec
    # run time with high-mem env and cache_size 1600 - 382  sec
    # run time with default env and cache_size 800 -  sec
    # run time with high-mem env and cache_size 800 - 407 sec
    # run time with high-mem env and cache_size 2400 - 446 sec  
    svm = SVC(kernel='linear', gamma=10, random_state=my_random_state, probability=True, cache_size=1600)
    #gd = GridSearchCV(estimator=svm, param_grid=parameters, cv=5)
    #gd.fit(x_train, y_train)
    #print(gd.best_params_)

    svm.fit(x_train, y_train)

    y_pred = svm.predict(x_test)
    confmat = confusion_matrix(y_true=y_test, y_pred=y_pred)
    print('svm w linear kernel')
    print(confmat)

    y_pred = svm.predict_proba(x_test)[:, 1]
    print('ROC_AUC_SCORE: ', roc_auc_score(y_true=y_test, y_score=y_pred))

    svm_disp = plot_roc_curve(svm, x_test, y_test)
    plt.show()

    stop = timeit.default_timer()
    print('Time: ', stop - start)  

@transform_pandas(
    Output(rid="ri.foundry.main.dataset.e0fd8f16-a131-4276-84c7-acc20e7f1829"),
    data_scaled_and_outcomes=Input(rid="ri.foundry.main.dataset.b474df3d-909d-4a81-9e38-515e22b9cff3"),
    inpatient_scaled_w_imputation=Input(rid="ri.foundry.main.dataset.f410db35-59e0-4b82-8fa8-d6dc6a61c9f2"),
    outcomes=Input(rid="ri.foundry.main.dataset.3d9b1654-3923-484f-8db5-6b38b56e290c")
)
def various_lr(data_scaled_and_outcomes, inpatient_scaled_w_imputation, outcomes):
    data_and_outcomes = data_scaled_and_outcomes

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

