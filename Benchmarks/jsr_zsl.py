import numpy as np
import argparse
import os
from scipy import io, spatial, linalg
from sklearn.metrics import confusion_matrix
import xlwt

parser = argparse.ArgumentParser(description="JSR")

# datasets setting
parser.add_argument('-data', '--dataset', default='AWA1', type=str)

'''
AWA1 >> Acc:0.674
AWA2 >> Acc:0.661
CUB  >> Acc:0.513
SUN  >> Acc:0.615
aPY  >> Acc:0.303
'''

class JSR():

    def __init__(self, args):

        self.args = args

        data_folder = os.getcwd() + '/data/' + args.dataset + '/'
        res101 = io.loadmat(data_folder + 'res101.mat')
        att_splits = io.loadmat(data_folder + 'att_splits.mat')

        # ======X data: features===========
        train_loc = 'trainval_loc'
        val_loc = 'val_loc'
        test_loc = 'test_unseen_loc'

        feat = res101['features']
        self.X_train = feat[:, np.squeeze(att_splits[train_loc] - 1)]
        self.X_val = feat[:, np.squeeze(att_splits[val_loc] - 1)]
        self.X_test = feat[:, np.squeeze(att_splits[test_loc] - 1)]


        # ======label data: labels===========
        labels = res101['labels']
        self.labels_train = labels[np.squeeze(att_splits[train_loc] - 1)]
        # val is not be used
        self.labels_val = labels[np.squeeze(att_splits[val_loc] - 1)]
        self.labels_test = labels[np.squeeze(att_splits[test_loc] - 1)]

        # seen class label
        train_labels_seen = np.unique(self.labels_train)
        # val is not be used
        val_labels_unseen = np.unique(self.labels_val)
        # unseen class label
        test_labels_unseen = np.unique(self.labels_test)

        i = 0
        for labels in train_labels_seen:
            self.labels_train[self.labels_train == labels] = i
            i += 1

        j = 0
        for labels in val_labels_unseen:
            self.labels_val[self.labels_val == labels] = j
            j += 1

        k = 0
        for labels in test_labels_unseen:
            self.labels_test[self.labels_test == labels] = k
            k += 1

        sig = att_splits['att']  # k x C
        self.train_sig = sig[:, train_labels_seen - 1]
        self.val_sig = sig[:, val_labels_unseen - 1]
        self.test_sig = sig[:, test_labels_unseen - 1]

        # the semantic information of 20 seen classes in training data
        self.train_att = np.zeros((self.X_train.shape[1], self.train_sig.shape[0]))
        for i in range(self.train_att.shape[0]):
            self.train_att[i] = self.train_sig.T[self.labels_train[i][0]]

        self.X_train = self.normalizeFeature(self.X_train.T).T

    def normalizeFeature(self, x):
        # x = N x d (d:feature dimension, N:number of instances)
        x = x + 1e-10
        feature_norm = np.sum(x ** 2, axis=1) ** 0.5  # l2-norm
        feat = x / feature_norm[:, np.newaxis]

        return feat

    # JSR_function
    def find_W(self, X, S, alpha, beta, ld, P):
        # A = np.dot((beta+1), np.dot(S, S.T))
        A = (beta + 1) * np.dot(S, S.T)
        row = A.shape[0]
        col = A.shape[1]
        temp = np.eye(row, col)
        A = A + ld * temp

        # B = np.dot((alpha+1), np.dot(X, X.T))
        B = (alpha + 1) * np.dot(X, X.T)
        C = 2 * np.dot(S, X.T)
        W = linalg.solve_sylvester(A, B, C)

        return W

    def pca(self, X, d):
        # Centralization
        means = np.mean(X, 0)
        X = X - means
        # Covariance Matrix
        covM = np.dot(X.T, X)
        eigval, eigvec = np.linalg.eig(covM)
        indexes = np.argsort(eigval)[-d:]
        W = eigvec[:, indexes]
        return W

    # compute accuracy
    def zsl_acc(self, X, W, y_true, sig, mode):  # Class Averaged Top-1 Accuarcy

        if mode == 'F2S':
            # [F --> S], projecting data from feature space to semantic space
            F2S = np.dot(X.T, self.normalizeFeature(W).T)  # N x k
            dist_F2S = 1 - spatial.distance.cdist(F2S, sig.T, 'cosine')  # N x C(no. of classes)
            pred_F2S = np.array([np.argmax(y) for y in dist_F2S])
            cm_F2S = confusion_matrix(y_true, pred_F2S)
            cm_F2S = cm_F2S.astype('float') / cm_F2S.sum(axis=1)[:, np.newaxis]
            acc_F2S = sum(cm_F2S.diagonal()) / sig.shape[1]

            return acc_F2S

        if mode == 'S2F':
            # [S --> F], projecting from semantic to visual space
            S2F = np.dot(sig.T, self.normalizeFeature(W))
            dist_S2F = 1 - spatial.distance.cdist(X.T, self.normalizeFeature(S2F), 'cosine')
            pred_S2F = np.array([np.argmax(y) for y in dist_S2F])
            cm_S2F = confusion_matrix(y_true, pred_S2F)
            cm_S2F = cm_S2F.astype('float') / cm_S2F.sum(axis=1)[:, np.newaxis]
            acc_S2F = sum(cm_S2F.diagonal()) / sig.shape[1]

            return acc_S2F

        if mode == 'val':
            # [F --> S], projecting data from feature space to semantic space
            F2S = np.dot(X.T, self.normalizeFeature(W).T)  # N x k
            dist_F2S = 1 - spatial.distance.cdist(F2S, sig.T, 'cosine')  # N x C(no. of classes)
            # [S --> F], projecting from semantic to visual space
            S2F = np.dot(sig.T, self.normalizeFeature(W))
            dist_S2F = 1 - spatial.distance.cdist(X.T, self.normalizeFeature(S2F), 'cosine')

            pred_F2S = np.array([np.argmax(y) for y in dist_F2S])
            pred_S2F = np.array([np.argmax(y) for y in dist_S2F])

            cm_F2S = confusion_matrix(y_true, pred_F2S)
            cm_F2S = cm_F2S.astype('float') / cm_F2S.sum(axis=1)[:, np.newaxis]

            cm_S2F = confusion_matrix(y_true, pred_S2F)
            cm_S2F = cm_S2F.astype('float') / cm_S2F.sum(axis=1)[:, np.newaxis]

            acc_F2S = sum(cm_F2S.diagonal()) / sig.shape[1]
            acc_S2F = sum(cm_S2F.diagonal()) / sig.shape[1]

            # acc = acc_F2S if acc_F2S>acc_S2F else acc_S2F

            return acc_F2S, acc_S2F

    def evaluate(self):
        alph = 10
        bet = 20
        lambd = 10060

        P = self.pca(self.X_train, 85)
        best_W_S2F = self.find_W(self.X_train, self.train_att.T, alph, bet, lambd, P)

        # X_te，W, label_te, S_te,compute accuracy acc
        test_acc_S2F = self.zsl_acc(self.X_test, best_W_S2F, self.labels_test, self.test_sig, 'S2F')

        print('Alpha:{}, beta:{}, lambda:{}'.format(alph, bet, lambd))
        print('Acc:{}'.format(test_acc_S2F))


if __name__ == '__main__':

    args = parser.parse_args()
    print('Dataset : {}\n'.format(args.dataset))
    jsr = JSR(args)
    jsr.evaluate()
