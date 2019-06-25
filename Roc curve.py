import tensorflow as tf
import pickle
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve
from sklearn.metrics import auc


class RocCurve:

    image_width = 100
    image_height = 100

    @staticmethod
    def draw_roc():
        """
        load Cnn models
         """
        read_jason = open('vgg16_model/model_25.json')
        read_model = read_jason.read()
        read_jason.close()
        cnn_model = tf.keras.models.model_from_json(read_model)
        cnn_model.load_weights("vgg16_model/CNN_25.h5")
        print("Cnn model with 25 epochs successfully loaded")

        read_jason2 = open('vgg16_model/model_50.json')
        read_model2 = read_jason2.read()
        read_jason2.close()
        cnn_model2 = tf.keras.models.model_from_json(read_model2)
        cnn_model2.load_weights("vgg16_model/CNN_50.h5")
        print("Cnn model with 50 epochs successfully loaded")

        read_jason3 = open('vgg16_model/model_75.json')
        read_model3 = read_jason3.read()
        read_jason3.close()
        cnn_model3 = tf.keras.models.model_from_json(read_model3)
        cnn_model3.load_weights("vgg16_model/CNN_75.h5")
        print("Cnn model with 75 epochs successfully loaded")

        read_jason4 = open('vgg16_model/model_100.json')
        read_model4 = read_jason4.read()
        read_jason4.close()
        cnn_model4 = tf.keras.models.model_from_json(read_model4)
        cnn_model4.load_weights("vgg16_model/CNN_100.h5")
        print("Cnn model with 100 epochs successfully loaded")

        X = pickle.load(open("Vgg16_model/Images.pickle", "rb"))
        Y = pickle.load(open("Vgg16_model/Labels.pickle", "rb"))
        cnn_pred = cnn_model.predict(X)
        cnn_pred2 = cnn_model2.predict(X)
        cnn_pred3 = cnn_model3.predict(X)
        cnn_pred4 = cnn_model4.predict(X)

        """
        prediction
        """

        fp_cnn, tp_cnn, th_cnn = roc_curve(Y, cnn_pred)
        area_under_cnn = auc(fp_cnn, tp_cnn)

        fp_cnn2, tp_cnn2, th_cnn2 = roc_curve(Y, cnn_pred2)
        area_under_cnn2 = auc(fp_cnn2, tp_cnn2)

        fp_cnn3, tp_cnn3, th_cnn3 = roc_curve(Y, cnn_pred3)
        area_under_cnn3 = auc(fp_cnn3, tp_cnn3)

        fp_cnn4, tp_cnn4, th_cnn4 = roc_curve(Y, cnn_pred4)
        area_under_cnn4 = auc(fp_cnn4, tp_cnn4)

        """Plot the curve
        """
        fig = plt.figure()
        plt.plot([0, 1], [0, 1], 'k--')
        plt.plot(fp_cnn, tp_cnn, label='CNN_25 (area under curve = {:.3f})'.format(area_under_cnn))
        plt.plot(fp_cnn2, tp_cnn2, label='CNN_50 (area under curve = {:.3f})'.format(area_under_cnn2))
        plt.plot(fp_cnn3, tp_cnn3, label='CNN_75 (area under curve = {:.3f})'.format(area_under_cnn3))
        plt.plot(fp_cnn4, tp_cnn4, label='CNN_100 (area under curve = {:.3f})'.format(area_under_cnn4))
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC Curve')
        plt.legend(loc='best')
        plt.show()
        fig.savefig("vgg16_model/Roc Curve", dpi=fig.dpi)


v = RocCurve()
v.draw_roc()
