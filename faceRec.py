# Dependencies
# Please install
# 'CVXOPT' library for python used to solve quadratic equations
# 'sklearn' library used for shuffling the data and cross_validation
# 'scipy' library used for reading images
# 'numpy' library used for various calculations

# References ->
# https://www.youtube.com/user/sentdex/videos
# http://cvxopt.org/examples/tutorial/qp.html
# https://pythonprogramming.net/soft-margin-kernel-cvxopt-svm-machine-learning-tutorial/
# https://pythonprogramming.net/coding-k-nearest-neighbors-machine-learning-tutorial/
# http://stackoverflow.com/questions/13224362/principal-component-analysis-pca-in-python
# http://sebastianraschka.com/Articles/2014_python_lda.html
# http://stackoverflow.com for various syntax doubts.

import os
import numpy as np
import cvxopt
import scipy.misc
import math
from matplotlib import pyplot
from sklearn.utils import shuffle
from sklearn import cross_validation


class Support_Vector_Machine():

    def __init__(self):
        # Defining global variables
        self.c = 100.0
        self.lm = None
        self.sv_X = None
        self.sv_y = None
        self.b = None
        self.w = None

    def train(self, x, y):
        # Extracting sample and feature lengths
        sample_len, feature_len = x.shape

        # Generating Gramian Matrix
        M = np.zeros((sample_len, sample_len))
        for i in range(sample_len):
            for j in range(sample_len):
                M[i,j] = np.dot(x[i], x[j])

        # Calculating values of P, q, A, b, G, h
        P = cvxopt.matrix(np.outer(y, y) * M)
        q = cvxopt.matrix(np.ones(sample_len) * -1)
        A = cvxopt.matrix(y, (1, sample_len), 'd')
        b = cvxopt.matrix(0.0)
        G = cvxopt.matrix(np.vstack((np.diag(np.ones(sample_len) * -1), np.identity(sample_len))))
        h = cvxopt.matrix(np.hstack((np.zeros(sample_len), np.ones(sample_len) * self.c)))

        # Solving quadratic equation
        # You can set 'show_progress' to True to see cvxopt output
        cvxopt.solvers.options['show_progress'] = False
        sol = cvxopt.solvers.qp(P, q, G, h, A, b)
        lm = np.ravel(sol['x'])

        # Determining support vectors
        y = np.asarray(y)
        sv = lm > 0.0e-7
        index = np.arange(len(lm))[sv]
        self.lm = lm[sv]
        self.sv_X = x[sv]
        self.sv_y = y[sv]

        # Calculating bias
        self.b = 0.0
        for i in range(len(self.lm)):
            self.b = self.b + self.sv_y[i] - np.sum(self.lm * self.sv_y * M[index[i],sv])
        self.b /= len(self.lm)

        # Calculating weights
        self.w = np.zeros(feature_len)
        for i in range(len(self.lm)):
            self.w += self.lm[i] * self.sv_y[i] * self.sv_X[i]

    def predict(self, x):
        return np.sign(np.dot(x, self.w) + self.b)


def PCA(X_train, X_test, k):
    # Centerize the images
    X_train -= np.mean(X_train, axis=0)
    X_test -= np.mean(X_test, axis=0)

    print 'Calculating Covariance matrix'
    CovM = np.cov(X_train.T)

    print 'Calculating eigen values and eigen vectors, please wait...'
    evals, evecs = np.linalg.eigh(CovM)
    # Sort the eigen values in descending order and then sorted the eigen vectors by the same index
    idx = np.argsort(evals)[::-1][:k]
    evecs = evecs[:, idx]

    # Can uncomment for plotting eigen values graph
    # evals = evals[idx]
    # pyplot.plot(evals)
    # pyplot.show()
    return np.dot(evecs.T, X_train.T).T, np.dot(evecs.T, X_test.T).T


def LDA(X_train, y_train, X_test, k):

    print 'Calculating class wise mean vectors'
    m , n = X_train.shape
    class_wise_mean = []
    for i in range(1,41):
        idx = np.where(y_train==i)
        class_wise_mean.append(np.mean(X_train[idx], axis=0))

    print 'Calculating within-class scatter matrix'
    within_SM = np.zeros((n, n))
    for i, mean_vector in zip(range(1, 41), class_wise_mean):
        class_wise_M = np.zeros((n, n))
        idx = np.where(y_train==i)
        for img in X_train[idx]:
            img, mean_vector = img.reshape(n, 1), mean_vector.reshape(n, 1)
            class_wise_M += (img - mean_vector).dot((img - mean_vector).T)
        within_SM += class_wise_M

    print 'Calculating between-class scatter matrix'
    total_mean = np.mean(X_train, axis=0)
    between_SM = np.zeros((n, n))
    for i, mean_vector in enumerate(class_wise_mean):
        idx = np.where(y_train==i+1)
        cnt = X_train[idx].shape[0]
        mean_vector = mean_vector.reshape(n, 1)
        total_mean = total_mean.reshape(n, 1)
        between_SM += cnt * (mean_vector - total_mean).dot((mean_vector - total_mean).T)

    print 'Calculating eigen values and eigen vectors, please wait...'
    evals, evecs = np.linalg.eigh(np.linalg.inv(within_SM).dot(between_SM))
    idx = np.argsort(evals)[::-1][:k]
    evecs = evecs[:, idx]

    # Can uncomment for plotting eigen values graph
    # evals = evals[idx]
    # pyplot.plot(evals)
    # pyplot.show()
    return np.dot(evecs.T, X_train.T).T, np.dot(evecs.T, X_test.T).T


def kNN(X_train, y_train, X_test, y_test):
    total_corr = 0
    i = 0
    for clas in y_test:
        # for every image in test set compute
        predict = X_test[i]
        distances = []
        j = 0
        for group in y_train:
            # Calculate the euclidean distance between test image with every train image
            features = X_train[j]
            euclidean_distance = np.linalg.norm(np.subtract(features, predict))
            distances.append([euclidean_distance, group])
            j += 1

        # Sorte the distance in ascending order and take the 1st one as the result
        result = [k[1] for k in sorted(distances)[:1]]
        predicted = result[0]
        correct = clas

        # Compare the result with original class and computed the accuracy
        c = np.sum(predicted == correct)
        total_corr += c
        i += 1
    print 'Accuracy is ', float(total_corr)/80*100
    return float(total_corr)/80*100


if __name__ == "__main__":
    def img_input(resize=False):
        X,y = [], []
        path, direc, docs = os.walk("orl_faces").next()
        direc.sort()
        # Iterating through each subject
        for subject in direc:
            files = os.listdir(path+'/'+subject)
            for file in files:
                img = scipy.misc.imread(path+'/'+subject+'/'+file).astype(np.float32)
                if resize:
                    img = scipy.misc.imresize(img, (56, 46)).astype(np.float32)
                X.append(img.reshape(-1))
                y.append(int(subject[1:]))
        X = np.asarray(X)
        y = np.asarray(y)
        return X, y

    def cross_val(X, y, DR=None, Alg=None, All=False):
        X, y = shuffle(X, y)
        kf = cross_validation.KFold(len(y), n_folds=5)
        avg_acc_p = []
        avg_acc_l = []
        avg_acc_pl = []
        avg_acc_svm = []
        avg_acc_svm_pca = []
        fld = 1
        for train, test in kf:
            X_train, X_test, y_train, y_test = X[train], X[test], y[train], y[test]

            if DR == 'PCA' and Alg == '1NN' or All:
                print '\nRunning Fold', fld, 'for PCA + 1NN'
                X_train_pca, X_test_pca = PCA(X_train, X_test, 70)
                avg_acc_p.append(kNN(X_train_pca, y_train, X_test_pca, y_test))
                if not All:
                    fld += 1

            if DR == 'LDA' and Alg == '1NN' or All:
                print '\nRunning Fold', fld, 'for LDA + 1NN'
                X_train_lda, X_test_lda = LDA(X_train, y_train, X_test, 70)
                avg_acc_l.append(kNN(X_train_lda, y_train, X_test_lda, y_test))
                if not All:
                    fld += 1

            if DR == 'PCA+LDA' and Alg == '1NN' or All:
                print '\nRunning Fold', fld, 'for PCA + LDA + 1NN'
                X_train_pca, X_test_pca = PCA(X_train, X_test, 70)
                X_train_pca_lda, X_test_pca_lda = LDA(X_train_pca, y_train, X_test_pca, 25)
                avg_acc_pl.append(kNN(X_train_pca_lda, y_train, X_test_pca_lda, y_test))
                if not All:
                    fld += 1

            if DR is None and Alg == 'SVM' or All:
                print '\nRunning Fold', fld, 'for SVM'
                # Defining svm model
                svm = Support_Vector_Machine()
                y_train_ovr = [None]*len(y_train)
                y_test_ovr = [None]*len(y_test)
                accuracies = 0
                print 'Running SVM. Please wait...'
                for i in range(1, 41):
                    # Setting the selected class as '1' and rest as '-1' depicting the One vs Rest classification.
                    for j in range(0, 320):
                        if y_train[j] == (i):
                            y_train_ovr[j] = 1
                        else:
                            y_train_ovr[j] = -1
                    for j in range(0, 80):
                        if y_test[j] == (i):
                            y_test_ovr[j] = 1
                        else:
                            y_test_ovr[j] = -1

                    # Taking Set_A as training set and Set_B for testing
                    svm.train(X_train, y_train_ovr)
                    predict_class = svm.predict(X_test)
                    c = np.sum(predict_class == y_test_ovr)
                    accuracies += float(c)/len(predict_class)*100
                accuracy = math.ceil(accuracies/40)
                print 'Accuracy is ', accuracy
                avg_acc_svm.append(accuracy)
                if not All:
                    fld += 1

            if DR == 'PCA' and Alg == 'SVM' or All:
                print '\nRunning Fold', fld, 'for PCA + SVM'
                X_train_pca, X_test_pca = PCA(X_train, X_test, 70)
                svm = Support_Vector_Machine()
                y_train_ovr = [None]*len(y_train)
                y_test_ovr = [None]*len(y_test)
                accuracies_pca = 0
                print 'Running SVM. Please wait...'
                for i in range(1, 41):
                    # Setting the selected class as '1' and rest as '-1' depicting the One vs Rest classification.
                    for j in range(0, 320):
                        if y_train[j] == (i):
                            y_train_ovr[j] = 1
                        else:
                            y_train_ovr[j] = -1
                    for j in range(0, 80):
                        if y_test[j] == (i):
                            y_test_ovr[j] = 1
                        else:
                            y_test_ovr[j] = -1

                    # Taking Set_A as training set and Set_B for testing
                    svm.train(X_train_pca, y_train_ovr)
                    predict_class = svm.predict(X_test_pca)
                    c = np.sum(predict_class == y_test_ovr)
                    accuracies_pca += math.ceil(float(c)/len(predict_class)*100)
                accuracy = accuracies_pca/40.0
                print 'Accuracy is ', accuracy
                avg_acc_svm_pca.append(accuracy)
                if not All:
                    fld += 1
            if All:
                fld += 1

            print '\n'
        if DR == 'PCA' and Alg == '1NN' or All:
            print 'Average accuracy for PCA + 1NN ', sum(avg_acc_p)/5.0, '\n'

        if DR == 'LDA' and Alg == '1NN' or All:
            print 'Average accuracy for LDA + 1NN', sum(avg_acc_l)/5.0, '\n'

        if DR == 'PCA+LDA' and Alg == '1NN' or All:
            print 'Average accuracy for PCA + LDA + 1NN', sum(avg_acc_pl)/5.0, '\n'

        if DR is None and Alg == 'SVM' or All:
            print 'Average accuracy for SVM ', sum(avg_acc_svm)/5.0, '\n'

        if DR == 'PCA' and Alg == 'SVM' or All:
            print 'Average accuracy for PCA + SVM ', sum(avg_acc_svm_pca)/5.0, '\n'


    def task_selector():
        print '\n'
        choice = int(raw_input('1: PCA + 1NN \n'
                               '2: Resized Images + PCA + 1NN \n'
                               '3: LDA + 1NN \n'
                               '4: PCA + LDA + 1NN \n'
                               '5: SVM \n'
                               '6: PCA + SVM \n'
                               '7: Run all tasks \n'
                               '0: To quit \n'
                               'Please enter your choice: '))
        print '\n'
        if choice == 1:
            print "Running Task 1: PCA + 1NN"
            X, y = img_input()
            cross_val(X, y, DR='PCA', Alg='1NN')
            task_selector()

        elif choice == 2:
            print "Running Task 2: Resized Images + PCA + 1NN"
            X, y = img_input(resize=True)
            cross_val(X, y, DR='PCA', Alg='1NN')
            task_selector()

        elif choice == 3:
            print "Running Task 3: LDA + 1NN"
            X, y = img_input()
            cross_val(X, y, DR='LDA', Alg='1NN')
            task_selector()

        elif choice == 4:
            print "Running Task 4: PCA + LDA + 1NN"
            X, y = img_input()
            cross_val(X, y, DR='PCA+LDA', Alg='1NN')
            task_selector()

        elif choice == 5:
            print "Running Task 5: SVM\n"
            X, y = img_input()
            cross_val(X, y, DR=None, Alg='SVM')
            task_selector()

        elif choice == 6:
            print "Running Task 6: PCA + SVM"
            X, y = img_input()
            cross_val(X, y, DR='PCA', Alg='SVM')
            task_selector()

        elif choice == 7:
            print "Running All Tasks"

            print "Running Task 2: Resized Images + PCA + 1NN\n"
            X, y = img_input(resize=True)
            cross_val(X, y, DR='PCA', Alg='1NN')

            print "Running Rest of the Tasks, from 1,3-6 "
            X, y = img_input()
            cross_val(X, y, All=True)

            task_selector()

        elif choice == 0:
            quit()

        else:
            print 'Please enter a valid choice'
            task_selector()

    task_selector()
