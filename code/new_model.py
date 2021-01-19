# -*-encoding:utf-8 -*-
import numpy as np
from get_sample import get_train_data
from sklearn.model_selection import KFold
from sklearn.utils import shuffle
from sklearn import metrics
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LinearRegression
from sklearn.metrics import precision_score, recall_score, f1_score
from sklearn.metrics import roc_curve, auc
from sklearn.metrics import precision_recall_curve
from numpy import interp


mirna_fun_dict_path = "../data/miRNA_fun_dict.npy"
mirna_ga_dict_path = '../data/miRNA_ga_dict.npy'
disease_model1_dict_path = '../data/dis_model1_dict.npy'
disease_ga_dict_path = "../data/dis_ga_dict.npy"
mirna_fun_dict = np.load(mirna_fun_dict_path, allow_pickle=True).item()
mirna_ga_dict = np.load(mirna_ga_dict_path, allow_pickle=True).item()
disease_model1_dict = np.load(disease_model1_dict_path, allow_pickle=True).item()
disease_ga_dict = np.load(disease_ga_dict_path, allow_pickle=True).item()


def get_2d_data(data_index):
    ver1_list = []
    ver2_list = []
    ver3_list = []
    ver4_list = []
    for index in data_index:
        mirna = index[0]
        disease = index[1]
        ver1 = mirna_fun_dict[mirna].tolist() + disease_model1_dict[disease].tolist()
        ver2 = mirna_fun_dict[mirna].tolist() + disease_ga_dict[disease].tolist()
        ver3 = mirna_ga_dict[mirna].tolist() + disease_model1_dict[disease].tolist()
        ver4 = mirna_ga_dict[mirna].tolist() + disease_ga_dict[disease].tolist()
        ver1_list.append(ver1)
        ver2_list.append(ver2)
        ver3_list.append(ver3)
        ver4_list.append(ver4)
    return np.array(ver1_list), np.array(ver2_list), np.array(ver3_list), np.array(ver4_list)


def get_train_data_index():
    sample_data, label = get_train_data()
    sample_data = np.array(sample_data)
    label = np.array(label)
    return sample_data, label


def train():
    X, Y = get_train_data_index()
    X, Y = shuffle(X, Y, random_state=1)
    kf = KFold(n_splits=5)
    AUC_list = []
    p_list = []
    r_list = []
    f1_list = []
    fpr_list = []
    tpr_list = []
    AUPR_list = []
    tprs = []
    n_trees = 300
    mean_fpr = np.linspace(0, 1, 100)
    clf1 = RandomForestClassifier(n_estimators=n_trees, criterion='gini', random_state=1)
    clf2 = RandomForestClassifier(n_estimators=n_trees, criterion='gini', random_state=1)
    clf3 = RandomForestClassifier(n_estimators=n_trees,  criterion='gini', random_state=1)
    clf4 = RandomForestClassifier(n_estimators=n_trees,  criterion='gini', random_state=1)
    for train_index, test_index in kf.split(X, Y):
        X_train = X[train_index]
        Y_train = Y[train_index]
        X_test = X[test_index]
        Y_test = Y[test_index]
        ver1_list, ver2_list, ver3_list, ver4_list = get_2d_data(X_train)
        clf1.fit(ver1_list, Y_train)
        clf2.fit(ver2_list, Y_train)
        clf3.fit(ver3_list, Y_train)
        clf4.fit(ver4_list, Y_train)
        train1_predict = clf1.predict_proba(ver1_list)[:, 1]
        train2_predict = clf2.predict_proba(ver2_list)[:, 1]
        train3_predict = clf3.predict_proba(ver3_list)[:, 1]
        train4_predict = clf4.predict_proba(ver4_list)[:, 1]
        train_X = []
        for i in range(len(ver1_list)):
            x = []
            x.append(train1_predict[i])
            x.append(train2_predict[i])
            x.append(train3_predict[i])
            x.append(train4_predict[i])
            train_X.append(x)
        train_X = np.array(train_X)
        test1_list, test2_list, test3_list, test4_list = get_2d_data(X_test)
        test1_predict = clf1.predict_proba(test1_list)[:, 1]
        test2_predict = clf2.predict_proba(test2_list)[:, 1]
        test3_predict = clf3.predict_proba(test3_list)[:, 1]
        test4_predict = clf4.predict_proba(test4_list)[:, 1]
        test_Y = []
        for i in range(len(test1_list)):
            y = []
            y.append(test1_predict[i])
            y.append(test2_predict[i])
            y.append(test3_predict[i])
            y.append(test4_predict[i])
            test_Y.append(y)
        test_Y = np.array(test_Y)
        bclf = LinearRegression()
        bclf.fit(train_X, Y_train)
        Y_test_predict = bclf.predict(test_Y)
        fpr, tpr, threshold = roc_curve(Y_test, Y_test_predict)
        precision, recall, _ = precision_recall_curve(Y_test, Y_test_predict)
        AUCPR = auc(recall, precision)
        AUPR_list.append(AUCPR)
        AUC = metrics.roc_auc_score(Y_test, Y_test_predict)
        AUC_list.append(AUC)
        p = precision_score(Y_test, Y_test_predict.round())
        p_list.append(p)
        r = recall_score(Y_test, Y_test_predict.round())
        r_list.append(r)
        f1 = f1_score(Y_test, Y_test_predict.round())
        f1_list.append(f1)
        fpr_list.append(fpr)
        tpr_list.append(tpr)
        tprs.append(interp(mean_fpr, fpr, tpr))
        tprs[-1][0] = 0.0
        print("the AUC of the model is ", AUC)
    mean_tpr = np.mean(tprs, axis=0)
    mean_tpr[-1] = 1.0
    print("the average of the AUC is ", sum(AUC_list) / len(AUC_list))
    print("the average of the AUPR is ", sum(AUPR_list) / len(AUPR_list))
    print("the average of p is ", sum(p_list) / len(p_list))
    print("the average of r is ", sum(r_list) / len(r_list))
    print("the average of f1 is ", sum(f1_list) / len(f1_list))
    return fpr_list, tpr_list, AUC_list, mean_fpr, mean_tpr


if __name__ == "__main__":
    fpr_list, tpr_list, AUC_list, mean_fpr, mean_tpr = train()