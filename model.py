from tensorflow.keras.models import model_from_json
import tensorflow as tf
import numpy as np
#
# config = tf.compat.v1.ConfigProto()
# config.gpu.options.per_process_gpu_memeory_fraction = 0.5
# session = tf.compat.v1.Session(config=config)

class FacialExpressionModel(object):
    EMOTIONS_LIST = ['Angry',"Disgust",'Fear','Happy','Neutral','Sad','Surpirse']
    def __init__(self, model_json_file, model_weights_file):
        with open(model_json_file, 'r') as json_file:
            loaded_model_json = json_file.read()
            self.loaded_model = model_from_json(loaded_model_json)
        self.loaded_model.load_weights(model_weights_file)
        #self.loaded_model.make_predict_function()
    def predict_emotion(self, img):
        self.preds = self.loaded_model.predict(img)

        return self.EMOTIONS_LIST[(np.argmax(self.preds))] 
        #return FacialExpressionModel.EMOTIONS_LIST(np.argmax(self.preds))
