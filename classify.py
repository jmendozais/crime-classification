from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.preprocessing import LabelBinarizer, StandardScaler

from sklearn.model_selection import KFold
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import log_loss

from matplotlib import pyplot as plt
import numpy as np

def to_categorical(y, nb_classes=None):
    if not nb_classes:
        nb_classes = np.max(y)+1
    Y = np.zeros((len(y), nb_classes))
    for i in range(len(y)):
        Y[i, y[i]] = 1.
    return Y

X = np.load('X.npy')
Y = np.load('Y.npy')
Y = Y.T[0]

#X = X[:400000]
#Y = Y[:400000]

print 'X shape {}, Y shape {}'.format(X.shape, Y.shape)
folds = StratifiedKFold(n_splits=5, shuffle=True).split(X, Y)
tmp = []
for fold in folds:
    tmp.append(fold)
folds = tmp

Y_categorical = to_categorical(Y)

#folds = KFold(len(Y), n_folds=5, shuffle=True)

clfs = [
        #GaussianNB(), 
        #LinearDiscriminantAnalysis(),
        #DecisionTreeClassifier(),
        #RandomForestClassifier(), 
        #AdaBoostClassifier(), 
        #KNeighborsClassifier(), 
        #MLPClassifier(), 
        SVC(kernel='rbf', probability=True), 
        GaussianProcessClassifier(warm_start=True),
        QuadraticDiscriminantAnalysis()
        ]

def eval_clfs():
    losses_mean = []
    losses_std = []
    clf_names = []
    for clf in clfs:
        print '\n'
        print clf
        losses = []
        i = 0
        classifier_name = str(clf).split('(')[0]
        clf_names.append(classifier_name[:5]) 
 
        for tr, te in folds:
            print 'Classifier {} fold {}'.format(classifier_name, i)
            i += 1
            le = LabelBinarizer()
            le.fit(Y[tr])
            Y_tr_categorical = le.transform(Y[tr]) 
            Y_te_categorical = le.transform(Y[te])

            scaler = StandardScaler()
            X_tr = scaler.fit_transform(X[tr])
            X_te = scaler.transform(X[te])

            if isinstance(clf, MLPClassifier):
                clf.fit(X_tr, Y_tr_categorical)
            else:
                clf.fit(X_tr, Y[tr])

            Y_te_pred = clf.predict_proba(X_te).astype(int)
            loss = log_loss(Y_te_categorical, Y_te_pred)
            losses.append(loss)

            from sklearn.metrics import confusion_matrix
            from sklearn.metrics import classification_report

            Y_te = np.argmax(Y_te_categorical, axis=1)
            Y_te_pred = np.argmax(Y_te_pred, axis=1)
            cm = confusion_matrix(Y_te, Y_te_pred)
            print classification_report(Y_te, Y_te_pred)
            plt.matshow(cm)
            plt.title('CM for {}, fold {}, loss {}'.format(classifier_name, i, loss))
            plt.colorbar()
            plt.savefig('outs/{}-fold-{}.pdf'.format(classifier_name, i))
            plt.clf()

        losses = np.array(losses)
        print '{} +- {}'.format(losses.mean(), losses.std())
        losses_mean.append(losses.mean())
        losses_std.append(losses.std())

    print len(losses_mean), len(clf_names), len(losses_std)
    xticks = np.arange(len(clf_names)).astype(float)
    plt.bar(xticks, losses_mean, yerr=losses_std)
    plt.title('Bars'.format(classifier_name, i, loss))
    plt.xticks(xticks + 0.5, clf_names, rotation='vertical')
    plt.savefig('outs/clfs_losses.pdf')
    plt.clf()

    return losses_mean, losses_std


if __name__ == '__main__':
    eval_clfs()
