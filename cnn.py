from PIL import Image
from tensorflow.keras.preprocessing import image
import numpy as np
from tensorflow.keras.applications.inception_v3 import preprocess_input
from tensorflow.keras.models import load_model
import tensorflow as tf

from global_variables import *
from training import *

class CNN():
    def __init__(self, path_to_model):
        self.config = tf.ConfigProto()
        self.config.gpu_options.allow_growth = True
        self.session = tf.Session(config=self.config)

        width = CONFIG["models"]["default"]["max_width"]
        height = CONFIG["models"]["default"]["max_height"]

        self.target_size = (width, height)
        self.categories = []
        self.nb_classes = 0
        self.path_to_model_weights = path_to_model

        self.model = load_model(self.path_to_model_weights)

    def actualiser(self, path_to_model):
        self.model = load_model(path_to_model)

    def predict(self, path):
        img = Image.open(path)
        if img.size != self.target_size:
            img = img.resize(self.target_size)
        x = image.img_to_array(img)
        x = np.expand_dims(x, axis=0)
        x = preprocess_input(x)
        preds = self.model.predict(x)
        preds_et_probas = {}
        print(preds[0])

        for categorie, proba in zip(self.categories, preds[0]):
            preds_et_probas[categorie] = proba
        liste_triee = sorted(preds_et_probas, key=preds_et_probas.__getitem__, reverse=True)
        suretee = max(preds[0]) * 100

        return liste_triee[0], suretee, img