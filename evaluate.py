from sklearn.metrics import classification_report, roc_auc_score

def evaluate_model(model, X_val, y_val):
    y_pred_prob = model.predict(X_val)
    y_pred = (y_pred_prob > 0.5).astype(int)
    print(classification_report(y_val, y_pred))
    print("ROC-AUC Score:", roc_auc_score(y_val, y_pred_prob))