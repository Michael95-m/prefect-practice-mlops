from sklearn.metrics import roc_auc_score, f1_score, precision_score, recall_score
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("my_logger")

def model_eval(y_true, y_pred, y_pred_prob, print_result=False):
    auc = roc_auc_score(y_true, y_pred_prob)
    f1 = f1_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)

    if print_result:
        logger.info("AUC is: {:.3f}".format(auc))
        logger.info("f1 score is: {:.3f}".format(f1))
        logger.info("Precision score is: {:.3f}".format(precision))
        logger.info("Recall score is: {:.3f}".format(recall))
    
    return auc, f1, precision, recall

def f1_eval(y_pred, dtrain):
    y_true = dtrain.get_label()
    y_pred = (y_pred > 0.5).astype(int)
    return 'f1', f1_score(y_true, y_pred)