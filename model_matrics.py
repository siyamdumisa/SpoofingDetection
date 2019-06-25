import tensorflow as tf
import pickle

from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.metrics import roc_auc_score
from sklearn.metrics import confusion_matrix


class ModelMetrics:

    @staticmethod
    def model_one():

        read_jason = open('vgg16_model/model_25.json')
        read_model = read_jason.read()
        read_jason.close()
        cnn_model = tf.keras.models.model_from_json(read_model)
        cnn_model.load_weights("vgg16_model/CNN_25.h5")
        print("Cnn model successfully loaded")

        X = pickle.load(open("Vgg16_model/Images.pickle", "rb"))
        Y = pickle.load(open("Vgg16_model/Labels.pickle", "rb"))

        # predict probabilities for test set
        cnn_pred = cnn_model.predict(X, verbose=0)

        # predict crisp classes for test set
        yhat_classes = cnn_model.predict_classes(X, verbose=0)
        # reduce to 1d array
        yhat_probs = cnn_pred[:, 0]
        yhat_classes = yhat_classes[:, 0]

        # accuracy: (tp + tn) / (p + n)
        accuracy = accuracy_score(Y, yhat_classes)
        print('Accuracy: %f' % accuracy)
        # precision tp / (tp + fp)
        precision = precision_score(Y, yhat_classes)
        print('Precision: %f' % precision)
        # recall: tp / (tp + fn)
        recall = recall_score(Y, yhat_classes)
        print('Recall: %f' % recall)
        # f1: 2 tp / (2 tp + fp + fn)
        f1 = f1_score(Y, yhat_classes)
        print('F1 score: %f' % f1)
        # ROC AUC
        auc = roc_auc_score(Y, yhat_probs)
        print('ROC AUC: %f' % auc)
        # confusion matrix
        matrix = confusion_matrix(Y, yhat_classes)
        print(matrix)

    @staticmethod
    def model_two():
        read_jason = open('vgg16_model/model_50.json')
        read_model = read_jason.read()
        read_jason.close()
        cnn_model = tf.keras.models.model_from_json(read_model)
        cnn_model.load_weights("vgg16_model/CNN_50.h5")
        print("Cnn model successfully loaded")

        X = pickle.load(open("Vgg16_model/Images.pickle", "rb"))
        Y = pickle.load(open("Vgg16_model/Labels.pickle", "rb"))

        # predict probabilities for test set
        cnn_pred = cnn_model.predict(X, verbose=0)

        # predict crisp classes for test set
        yhat_classes = cnn_model.predict_classes(X, verbose=0)
        # reduce to 1d array
        yhat_probs = cnn_pred[:, 0]
        yhat_classes = yhat_classes[:, 0]

        # accuracy: (tp + tn) / (p + n)
        accuracy = accuracy_score(Y, yhat_classes)
        print('Accuracy: %f' % accuracy)
        # precision tp / (tp + fp)
        precision = precision_score(Y, yhat_classes)
        print('Precision: %f' % precision)
        # recall: tp / (tp + fn)
        recall = recall_score(Y, yhat_classes)
        print('Recall: %f' % recall)
        # f1: 2 tp / (2 tp + fp + fn)
        f1 = f1_score(Y, yhat_classes)
        print('F1 score: %f' % f1)

        # ROC AUC
        auc = roc_auc_score(Y, yhat_probs)
        print('ROC AUC: %f' % auc)
        # confusion matrix
        matrix = confusion_matrix(Y, yhat_classes)
        print(matrix)

    @staticmethod
    def model_three():
        read_jason = open('vgg16_model/model_75.json')
        read_model = read_jason.read()
        read_jason.close()
        cnn_model = tf.keras.models.model_from_json(read_model)
        cnn_model.load_weights("vgg16_model/CNN_75.h5")
        print("Cnn model successfully loaded")

        X = pickle.load(open("Vgg16_model/Images.pickle", "rb"))
        Y = pickle.load(open("Vgg16_model/Labels.pickle", "rb"))

        # predict probabilities for test set
        cnn_pred = cnn_model.predict(X, verbose=0)

        # predict crisp classes for test set
        yhat_classes = cnn_model.predict_classes(X, verbose=0)
        # reduce to 1d array
        yhat_probs = cnn_pred[:, 0]
        yhat_classes = yhat_classes[:, 0]

        # accuracy: (tp + tn) / (p + n)
        accuracy = accuracy_score(Y, yhat_classes)
        print('Accuracy: %f' % accuracy)
        # precision tp / (tp + fp)
        precision = precision_score(Y, yhat_classes)
        print('Precision: %f' % precision)
        # recall: tp / (tp + fn)
        recall = recall_score(Y, yhat_classes)
        print('Recall: %f' % recall)
        # f1: 2 tp / (2 tp + fp + fn)
        f1 = f1_score(Y, yhat_classes)
        print('F1 score: %f' % f1)

        # ROC AUC
        auc = roc_auc_score(Y, yhat_probs)
        print('ROC AUC: %f' % auc)
        # confusion matrix
        matrix = confusion_matrix(Y, yhat_classes)
        print(matrix)

    @staticmethod
    def model_four():
        read_jason = open('vgg16_model/model_100.json')
        read_model = read_jason.read()
        read_jason.close()
        cnn_model = tf.keras.models.model_from_json(read_model)
        cnn_model.load_weights("vgg16_model/CNN_100.h5")
        print("Cnn model successfully loaded")

        X = pickle.load(open("Vgg16_model/Images.pickle", "rb"))
        Y = pickle.load(open("Vgg16_model/Labels.pickle", "rb"))

        # predict probabilities for test set
        cnn_pred = cnn_model.predict(X, verbose=0)

        # predict crisp classes for test set
        yhat_classes = cnn_model.predict_classes(X, verbose=0)
        # reduce to 1d array
        yhat_probs = cnn_pred[:, 0]
        yhat_classes = yhat_classes[:, 0]

        # accuracy: (tp + tn) / (p + n)
        accuracy = accuracy_score(Y, yhat_classes)
        print('Accuracy: %f' % accuracy)
        # precision tp / (tp + fp)
        precision = precision_score(Y, yhat_classes)
        print('Precision: %f' % precision)
        # recall: tp / (tp + fn)
        recall = recall_score(Y, yhat_classes)
        print('Recall: %f' % recall)
        # f1: 2 tp / (2 tp + fp + fn)
        f1 = f1_score(Y, yhat_classes)
        print('F1 score: %f' % f1)
        # ROC AUC
        auc = roc_auc_score(Y, yhat_probs)
        print('ROC AUC: %f' % auc)
        # confusion matrix
        matrix = confusion_matrix(Y, yhat_classes)
        print(matrix)


metrics = ModelMetrics()

print("--------------------------------------------------------------------------------------------------------")
print(" Cnn Model with 25 epochs")
metrics.model_one()
print('\n')

print("--------------------------------------------------------------------------------------------------------")
print(" Cnn Model with 50 epochs")
metrics.model_two()
print('\n')

print("--------------------------------------------------------------------------------------------------------")
print(" Cnn Model with 75 epochs")
metrics.model_three()
print('\n')

print("--------------------------------------------------------------------------------------------------------")
print(" Cnn Model with 100 epochs")
metrics.model_four()
print('\n')
