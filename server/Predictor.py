import numpy as np

from sklearn import preprocessing
from sklearn import model_selection
from sklearn.pipeline import make_pipeline

from sklearn import ensemble
from sklearn import gaussian_process
from sklearn import linear_model
from sklearn import neighbors
from sklearn import neural_network
from sklearn import svm
from sklearn import tree

import pickle

# For the PNNL Data:
# Gradient Boosting: 0.806777 +/- 0.096609
# GP: 0.092744 +/- 0.352183
# ARD: 0.192380 +/- 0.695032
# ElasticNet: 0.298822 +/- 0.445474
# Huber: 0.248975 +/- 0.656531
# LARS: 0.237592 +/- 0.633618
# LASSO: 0.238349 +/- 0.632020
# OLS: 0.237592 +/- 0.633618
# OMP: 0.169401 +/- 0.721100
# PAR: 0.023094 +/- 0.236122
# Ridge: 0.241813 +/- 0.624572
# KNN: 0.285822 +/- 0.503111
# Multi-layer Perceptron: -0.241078 +/- 0.226819


class Predictor(object):
    """ A predictor class that will try to build several types of models
        to fit a dataset. Each type of model will be cross-validated to
        attempt to fit the hyperparameters for the model. The best model
        (highest cv score) is retained for further prediction.
    """

    def fit(self, X, y):
        """ Function for selecting the best model to use on the given
            data. This method will try a bunch of different predictors
            from scikit-learn with various parameter settings and settle
            on the best for the given problem dataset
            @ In, X, a matrix of
        """
        # String name for plotting, Instance, Dictionary of parameters
        # to cross-validate
        testModels = [
                      ('AdaBoost', ensemble.AdaBoostRegressor(),
                       {'loss': ['linear', 'square', 'exponential'],
                        'n_estimators': [25, 50, 75, 100],
                        'learning_rate': [0.01, 0.05, 0.1, 0.5]}),
                      ('Bagging', ensemble.BaggingRegressor(),
                       {'n_estimators': [25, 50, 75, 100],
                        'max_samples': [0.25, 0.5, 0.75, 1.0],
                        'max_features': [0.25, 0.5, 0.75, 1.0],
                        'bootstrap_features': [False, True]}),
                      ('Ensemble ET', ensemble.ExtraTreesRegressor(), {}),
                      ('Gradient Boosting',
                       ensemble.GradientBoostingRegressor(),
                       {'loss': ['ls', 'lad', 'huber', 'quantile'],
                        'learning_rate': [0.01, 0.05, 0.1, 0.5],
                        'n_estimators': [25, 50, 75, 100, 250],
                        }),
                      ('Random Forest', ensemble.RandomForestRegressor(), {}),
                      ('GP', gaussian_process.GaussianProcessRegressor(),
                       {'n_restarts_optimizer': [0, 1, 5, 10],
                        'normalize_y': [False, True]}),
                      ('ARD', linear_model.ARDRegression(),
                       {'alpha_1': [1e-8, 1e-6, 1e-4],
                        'alpha_2': [1e-8, 1e-6, 1e-4],
                        'lambda_1': [1e-8, 1e-6, 1e-4],
                        'lambda_2': [1e-8, 1e-6, 1e-4],
                        'threshold_lambda': [1e2, 1e4, 1e6],
                        'fit_intercept': [False, True],
                        'normalize': [False, True]}),
                      ('ElasticNet', linear_model.ElasticNet(),
                       {'alpha': [0, 0.25, 0.5, 0.75, 1],
                        'l1_ratio': [0, 0.25, 0.5, 0.75, 1],
                        'fit_intercept': [False, True]}),
                      ('Huber', linear_model.HuberRegressor(), {}),
                      ('LARS', linear_model.Lars(), {}),
                      ('LASSO', linear_model.Lasso(), {}),
                      ('OLS', linear_model.LinearRegression(), {}),
                      ('OMP', linear_model.OrthogonalMatchingPursuit(), {}),
                      ('PAR', linear_model.PassiveAggressiveRegressor(), {}),
                      ('Ridge', linear_model.Ridge(), {}),
                      ('KNN', neighbors.KNeighborsRegressor(),
                       {'n_neighbors': [2, 5, 10, 15, 20, 25, 50],
                        'weights': ['uniform', 'distance']}),
                      # ('Radius Neighborhood',
                      #  neighbors.RadiusNeighborsRegressor(), {}),
                      ('Multi-layer Perceptron', neural_network.MLPRegressor(),
                       {'activation': ['identity', 'logistic', 'tanh', 'relu'],
                        'hidden_layer_sizes': [(100,), (10, 10), (1000,),
                                               (100, 2)],
                        'learning_rate': ['constant', 'invscaling',
                                          'adaptive']}),
                      ('SVR', svm.SVR(),
                       {'kernel': ['sigmoid', 'rbf', 'linear', 'poly'],
                        'degree': [2, 3],
                        'C': [1, 1e3, 1e6, 1e9],
                        'gamma': ['auto'],
                        'cache_size': [8000]}),
                      ('DT', tree.DecisionTreeRegressor(), {}),
                      ('ET', tree.ExtraTreeRegressor(), {})]

        # testModels = [('Gradient Boosting',
        #                ensemble.GradientBoostingRegressor(),
        #                {'loss': ['ls', 'lad', 'huber', 'quantile'],
        #                 'learning_rate': [0.01, 0.05, 0.1, 0.5],
        #                 'n_estimators': [25, 50, 75, 100, 250],
        #                 })]

        # Regression scoring techniques:
        # scoring = 'explained_variance'
        # scoring = 'neg_mean_absolute_error'
        # scoring = 'neg_mean_squared_error'
        # scoring = 'neg_mean_squared_log_error'
        # scoring = 'neg_median_absolute_error'
        scoring = 'r2'

        seed = 8

        best_model = None
        best_cv_score = -1

        X_norm = preprocessing.scale(X)

        for name, model, params in testModels:
            print('Parameter searching for ' + name)
            # Cross-validate the data

            cvModel = model_selection.GridSearchCV(model, params,
                                                   scoring=scoring,
                                                   fit_params=None, n_jobs=1,
                                                   iid=True, refit=True,
                                                   cv=None, verbose=0,
                                                   pre_dispatch='2*n_jobs',
                                                   error_score='raise',
                                                   return_train_score='warn')
            cvModel.fit(X_norm, y)
            best_fit = cvModel.best_params_
            bestimator = cvModel.best_estimator_

            print('Done')
            print(best_fit)
            print('Cross-validation score for best ' + name)

            kfold = model_selection.KFold(n_splits=10, random_state=seed)
            cv_results = model_selection.cross_val_score(bestimator,
                                                         X_norm, y,
                                                         cv=kfold,
                                                         scoring=scoring)
            print('Done')
            msg = "%s: %f +/- %f" % (name, cv_results.mean(), cv_results.std())
            print(msg)

            if cv_results.mean() > best_cv_score:
                best_cv_score = cv_results.mean()
                best_model = bestimator

            self.pipeline = make_pipeline(preprocessing.StandardScaler(),
                                          best_model)
            self.pipeline.fit(X, y)

    def predict(self, X):
        return self.pipeline.predict(X)

    def save(self, filename='model.pk'):
        pickle.dumps(self.pipeline, filename)

    def load(self, filename='model.pk'):
        self.pipeline = pickle.loads(filename)