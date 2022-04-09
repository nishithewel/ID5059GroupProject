"""A module that allows the stacking of models."""

from sklearn.ensemble import StackingClassifier
from sklearn.linear_model import LogisticRegression

def stack_models(*models, cv_folds=5, verbose=1):
    """Combines the given models using a stacking classifier.

    Args:
        models: tuple of (str, estimator).
          Base estimators which will be stacked together. Each element of the
          list is defined as a tuple of string (i.e. name) and an estimator
          instance. An estimator can be set to ‘drop’ using set_params.
        cv_folds, default=5: number of cv folds.
        verbose, default=1: verbosity level.

    Returns:
        A Stack of estimators with final classifier.
    
    Notes:
        For similar examples, see "Combine predictors using stacking"
          (Guillaume Lemaitre, Maria Telenczuk)
          https://runebook.dev/en/docs/scikit_learn/auto_examples/ensemble/plot_stack_predictors#sphx-glr-auto-examples-ensemble-plot-stack-predictors-py
          and "How To Use “Model Stacking” To Improve Machine Learning
          Predictions" (Trevor Pedersen)
          https://medium.com/geekculture/how-to-use-model-stacking-to-improve-machine-learning-predictions-d113278612d4#:~:text=Model%20Stacking%20is%20a%20way,model%20called%20a%20meta%2Dlearner.
          
    See also:
        sklearn.ensemble.StackingClassifier
    """
    # Create a list to store base models
    level0 = []
    
    for model in models:
        level0.append(model)

    # The "meta-learner" designated as the level1 model
    level1 = LogisticRegression()
    
    # Create the stacking ensemble
    model = StackingClassifier(estimators=level0,
                               final_estimator=level1,
                               cv=cv_folds,
                               verbose=verbose,
                               n_jobs = -1)
    return model