
from pandas import DataFrame
from sklearn.metrics import roc_curve, roc_auc_score, balanced_accuracy_score

def get_roc_auc(y_true, y_score):
    """Calculates the fpr, tpr and ROC-AUC.

    Args:
        y_true: ndarray of shape (n_samples,)
          True binary labels. Either {-1, 1} or {0, 1}.
        y_score : ndarray of shape (n_samples,)
          Target scores, probability estimates of the positive class.

    Returns:
        fpr: ndarray of shape (>2,)
          Increasing false positive rates. Corresponds to 1 - specificity.
        tpr: ndarray of shape (>2,)
          Increasing true positive rates. Corresponds to sensitivity.
        auc: float
          Area Under the Receiver Operating Characteristic Curve.
          
     See also:
        `sklearn.metrics.roc_auc_score`
        `sklearn.metrics.roc_curve`
    """
    fpr, tpr, _ = roc_curve(y_true, y_score)
    auc = roc_auc_score(y_true, y_score)
    return fpr, tpr, auc

def plot_multiple_ROC(*named_preds, y_true, X):
    rand_preds = [0] * len(y_true)
    rand_fpr, rand_tpr, rand_auc = get_roc_auc(y_true, rand_preds)
    plt.plot(rand_fpr, rand_tpr, '--', alpha = 0.4, label = f'Random Prediction (AUC = {rand_auc:0.2f})')
    
    aucs = []
    for name, pred in named_preds:
        fpr, tpr, auc = get_roc_auc(y_true, pred)
        aucs.append(auc)
        plt.plot(fpr, tpr, alpha = 0.4, label = f'{name} (AUC = {auc:0.3f})')
    
    plt.title("ROC Plot")
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.legend()
    plt.show()
    return aucs

def get_diagnostics(*models, y_true, X):
    preds = {name: model.predict_proba(X)[:,1] for name, model in models}
    aucs = plot_multiple_ROC(*preds, y_true=y_true, X=X)
    b_accuracy = [balanced_accuracy_score(y_true, pred) for pred in preds]
    
    model = preds.keys()
    return DataFrame(data={'model':model,
                           'auc':aucs,
                           'balanced accuracy': b_accuracy})
    
    
    
    