def classification_suitability_Parameters(x,y, x_train, y_train, x_test, y_test, classifier):
    
    # Methods / Parameters to check Suitability of Classification 
    from sklearn.metrics import (
        confusion_matrix, accuracy_score, precision_score, recall_score, f1_score,
        roc_auc_score, log_loss, matthews_corrcoef, cohen_kappa_score
    )

    y_train_predict = classifier.predict(x_train)
    y_predict = classifier.predict(x)
    y_test_predict = classifier.predict(x_test)

    # For classifiers that support predict_proba
    if hasattr(classifier, "predict_proba"):
        y_prob = classifier.predict_proba(x)
        y_test_prob = classifier.predict_proba(x_test)
        y_train_prob = classifier.predict_proba(x_train)
    else:
        y_prob = None
        y_test_prob = None
        y_train_prob = None

    # 1. Confusion Matrix
    cm = confusion_matrix(y, y_predict)
    cm_train = confusion_matrix(y_train, y_train_predict)
    cm_test = confusion_matrix(y_test, y_test_predict)

    print("\n\nConfusion Matrix (Whole Set) : \n", cm)
    print("Confusion Matrix (Training Set) : \n", cm_train)
    print("Confusion Matrix (Test Set) : \n", cm_test)

    # 2. Accuracy
    print("\nAccuracy (Whole Set) : ", accuracy_score(y, y_predict))
    print("Accuracy (Training Set) : ", accuracy_score(y_train, y_train_predict))
    print("Accuracy (Test Set) : ", accuracy_score(y_test, y_test_predict))

    # 3. Precision, Recall, F1 (macro avg for multi-class)
    print("\nPrecision (Whole Set) : ", precision_score(y, y_predict, average="macro"))
    print("Precision (Training Set) : ", precision_score(y_train, y_train_predict, average="macro"))
    print("Precision (Test Set) : ", precision_score(y_test, y_test_predict, average="macro"))

    print("\nRecall (Whole Set) : ", recall_score(y, y_predict, average="macro"))
    print("Recall (Training Set) : ", recall_score(y_train, y_train_predict, average="macro"))
    print("Recall (Test Set) : ", recall_score(y_test, y_test_predict, average="macro"))

    print("\nF1 Score (Whole Set) : ", f1_score(y, y_predict, average="macro"))
    print("F1 Score (Training Set) : ", f1_score(y_train, y_train_predict, average="macro"))
    print("F1 Score (Test Set) : ", f1_score(y_test, y_test_predict, average="macro"))

    # 4. Specificity (Only valid for binary classification)
    if cm.shape == (2, 2):
        tn, fp, fn, tp = cm.ravel()
        tn_tr, fp_tr, fn_tr, tp_tr = cm_train.ravel()
        tn_test, fp_test, fn_test, tp_test = cm_test.ravel()

        print("\nSpecificity (Whole Set) : ", tn / (tn + fp))
        print("Specificity (Training Set) : ", tn_tr / (tn_tr + fp_tr))
        print("Specificity (Test Set) : ", tn_test / (tn_test + fp_test))
    else:
        print("\nSpecificity: Not defined for multi-class problems")

    # 5. ROC-AUC (Only valid if predict_proba available)
    if y_prob is not None:
        if len(set(y)) == 2:  # binary
            print("\nROC-AUC (Whole Set) :", roc_auc_score(y, y_prob[:,1]))
            print("ROC-AUC (Training Set) :", roc_auc_score(y_train, y_train_prob[:,1]))
            print("ROC-AUC (Test Set) :", roc_auc_score(y_test, y_test_prob[:,1]))
        else:  # multi-class
            print("\nROC-AUC (Whole Set) :", roc_auc_score(y, y_prob, multi_class="ovr"))
            print("ROC-AUC (Training Set) :", roc_auc_score(y_train, y_train_prob, multi_class="ovr"))
            print("ROC-AUC (Test Set) :", roc_auc_score(y_test, y_test_prob, multi_class="ovr"))
    else:
        print("\nROC-AUC: Not available (classifier has no predict_proba)")

    # 6. Log Loss (if prob available)
    if y_prob is not None:
        print("\nLog Loss (Whole Set) :", log_loss(y, y_prob))
        print("Log Loss (Training Set) :", log_loss(y_train, y_train_prob))
        print("Log Loss (Test Set) :", log_loss(y_test, y_test_prob))
    else:
        print("\nLog Loss: Not available (classifier has no predict_proba)")

    # 7. Matthews Correlation Coefficient (MCC)
    print("\nMCC (Whole Set) :", matthews_corrcoef(y, y_predict))
    print("MCC (Training Set) :", matthews_corrcoef(y_train, y_train_predict))
    print("MCC (Test Set) :", matthews_corrcoef(y_test, y_test_predict))

    # 8. Cohen's Kappa
    print("\nCohen's Kappa (Whole Set) :", cohen_kappa_score(y, y_predict))
    print("Cohen's Kappa (Training Set) :", cohen_kappa_score(y_train, y_train_predict))
    print("Cohen's Kappa (Test Set) :", cohen_kappa_score(y_test, y_test_predict))