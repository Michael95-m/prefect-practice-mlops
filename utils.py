from sklearn.metrics import roc_auc_score, f1_score, precision_score, recall_score
import logging

logging.basicConfig(format='%(asctime)s - %(levelname)s - %(message)s', level=logging.INFO)

def model_eval(y_true, y_pred, y_pred_prob, print_result=False):
    auc = roc_auc_score(y_true, y_pred_prob)
    f1 = f1_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)

    if print_result:
        logging.info("AUC is: {:.3f}".format(auc))
        logging.info("f1 score is: {:.3f}".format(f1))
        logging.info("Precision score is: {:.3f}".format(precision))
        logging.info("Recall score is: {:.3f}".format(recall))
    
    return auc, f1, precision, recall

def f1_eval(y_pred, dtrain):
    y_true = dtrain.get_label()
    y_pred = (y_pred > 0.5).astype(int)
    return 'f1', f1_score(y_true, y_pred)