#!C:/Users/EnzoGamer/AppData/Local/conda/conda/envs/tf_gpu/python.exe
path = "C:/Users/EnzoGamer/AppData/Local/conda/conda/envs/tf_gpu/python.exe"
"""
Auteur : DUTRA Enzo (14/2/2019)
"""

import subprocess
import traceback
import trace 





from tkinter import *
from tkinter.filedialog import *
from tkinter import ttk as tkk
from tkinter.messagebox import showerror, showinfo
import tkinter as tk
from threading import Thread
from lxml import etree
import ntpath
import shutil
import matplotlib.pyplot as plt
#%matplotlib inline

from keras.layers import Dense, GlobalAveragePooling2D
from keras.models import Model
from PIL import Image
from keras.preprocessing import image
import numpy as np
from keras.applications.inception_v3 import preprocess_input

import sys
import os

RMSPROP_LR_DEFAULT = 0.001
ADAM_LR_DEFAULT = 0.001
BATCH_SIZE_DEFAULT = 16
IM_WIDTH, IM_HEIGHT = 299, 299 # fixed size for InceptionV3
NB_IV3_LAYERS_TO_FREEZE = 1 # nb de couches d'inception V3 à laisser statique

user_input = ''

def change_path(path):
    content = []
    with open(__file__,"r", encoding="ISO-8859-1") as f:
        for line in f:
            content.append(line)
    
    with open(__file__,"w", encoding="ISO-8859-1") as f:
        content[0] = "#!{n}\n".format(n=path)
        content[1] = "path = \"{n}\"\n".format(n=path)
        for i in range(len(content)):
            f.write(content[i])




# Idée de fonction du logiciel pour augmenter un dossier d'exemples
def run_dataset_configuration(data_dir, nb_images_augmentation, liste_options_augmentation):
    pass




def setup_to_transfer_learn(model, base_model, pas_preentrainement, optimiseur_preentrainement):
    from keras.optimizers import Adam, RMSprop, SGD
    """Freeze all layers and compile the model"""
    for layer in base_model.layers:
        layer.trainable = False
    if(optimiseur_preentrainement == "RMSprop"):
        optimiseur = RMSprop(lr=pas_preentrainement, rho=0.9, epsilon=None, decay=0.0)
        model.compile(optimizer=optimiseur, loss='categorical_crossentropy', metrics=['accuracy'])
    elif optimiseur_entrainement == "AMSGrad":
        optimiseur = Adam(lr=pas_entrainement, beta_1=0.9, beta_2=0.999, epsilon=1e-8, amsgrad=True)
        model.compile(optimizer=optimiseur, loss='categorical_crossentropy', metrics=['accuracy'])
    elif optimiseur_entrainement == "Adam":
        optimiseur = Adam(lr=pas_entrainement, beta_1=0.9, beta_2=0.999, epsilon=1e-8, amsgrad=False)
        model.compile(optimizer=optimiseur, loss='categorical_crossentropy', metrics=['accuracy'])
    elif optimiseur_preentrainement == "SGD":
        optimiseur = SGD(lr=pas_preentrainement, momentum=0.0, decay=0.0, nesterov=False)
        model.compile(optimizer=optimiseur, loss='categorical_crossentropy', metrics=['accuracy'])
    else:
        optimiseur = RMSprop(lr=pas_preentrainement, rho=0.9, epsilon=None, decay=0.0)
        model.compile(optimizer=optimiseur, loss='categorical_crossentropy', metrics=['accuracy'])



def add_new_last_layer(base_model, nb_classes):
    from keras.models import Model
    from keras.layers import Dense, GlobalAveragePooling2D
    """Add last layer to the convnet
    Args:
    base_model: keras model excluding top
    nb_classes: # of classes
    Returns:
    new keras model with last layer
    """
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    try:
        x = Dense(int(nb_classes), activation='relu')(x) #new FC layer, random init
    except:
        print("erreur, le type de nb_classes est : " + type(nb_classes))
        print("tentative de correction ...")
        try:
            x = Dense(int(nb_classes + 0), activation='relu')(x) #new FC layer, random init
        except:
            print("erreur, le type de nb_classes est : " + type(nb_classes))
            print("tentative de correction ...")
            try:
                x = Dense(float(nb_classes), activation='relu')(x) #new FC layer, random init
            except:
                print("erreur, le type de nb_classes est : " + type(nb_classes))
                print("tentative de correction ...")
                try:
                    nb_classes = str(nb_classes)
                    x = Dense(nb_classes, activation='relu')(x) #new FC layer, random init
                except:
                    print("erreur, le type de nb_classes est : " + type(nb_classes))
                    print("tentative de correction ...")
                    try:
                        x = Dense((int(nb_classes)+float(nb_classes))/2, activation='relu')(x) #new FC layer, random init
                    except:
                        print("erreur, le type de nb_classes est : " + type(nb_classes))
                        print("tentative de correction ...")
                        try:
                            x = Dense(long(nb_classes), activation='relu')(x) #new FC layer, random init
                        except:
                            print("erreur, le type de nb_classes est : " + type(nb_classes))
    try:
        predictions = Dense(int(nb_classes), activation='softmax')(x) #new FC layer, random init
    except:
        print("erreur, le type de nb_classes est : " + type(nb_classes))
        print("tentative de correction ...")
        try:
            predictions = Dense(int(nb_classes + 0), activation='softmax')(x) #new FC layer, random init
        except:
            print("erreur, le type de nb_classes est : " + type(nb_classes))
            print("tentative de correction ...")
            try:
                predictions = Dense(float(nb_classes), activation='softmax')(x) #new FC layer, random init
            except:
                print("erreur, le type de nb_classes est : " + type(nb_classes))
                print("tentative de correction ...")
                try:
                    nb_classes = str(nb_classes)
                    predictions = Dense(nb_classes, activation='softmax')(x) #new FC layer, random init
                except:
                    print("erreur, le type de nb_classes est : " + type(nb_classes))
                    print("tentative de correction ...")
                    try:
                        predictions = Dense((int(nb_classes)+float(nb_classes))/2, activation='softmax')(x) #new FC layer, random init
                    except:
                        print("erreur, le type de nb_classes est : " + type(nb_classes))
                        print("tentative de correction ...")
                        try:
                            predictions = Dense(long(nb_classes), activation='softmax')(x) #new FC layer, random init
                        except:
                            print("erreur, le type de nb_classes est : " + type(nb_classes))
    #predictions = Dense(nb_classes, activation='softmax')(x) #new softmax layer
    model = Model(input=base_model.input, output=predictions)
    return model


def setup_to_finetune(model, pas_entrainement, optimiseur_entrainement, nb_layers_to_freeze = 1, change_nb_l_to_f = True):
    from keras.optimizers import Adam, RMSprop, SGD
    """Freeze the bottom NB_IV3_LAYERS and retrain the remaining top layers.
    note: NB_IV3_LAYERS corresponds to the top 2 inception blocks in the inceptionv3 arch
    Args:
    model: keras model
    """

    if change_nb_l_to_f:
        for layer in model.layers[:nb_layers_to_freeze]:
            layer.trainable = False
        for layer in model.layers[nb_layers_to_freeze:]:
            layer.trainable = True
    
    if(optimiseur_entrainement == "RMSprop"):
        optimiseur = RMSprop(lr=pas_entrainement, rho=0.9, epsilon=None, decay=0.0)
        model.compile(optimizer=optimiseur, loss='categorical_crossentropy', metrics=['accuracy'])
    elif optimiseur_entrainement == "AMSGrad":
        optimiseur = Adam(lr=pas_entrainement, beta_1=0.9, beta_2=0.999, epsilon=1e-8, amsgrad=True)
        model.compile(optimizer=optimiseur, loss='categorical_crossentropy', metrics=['accuracy'])
    elif optimiseur_entrainement == "Adam":
        optimiseur = Adam(lr=pas_entrainement, beta_1=0.9, beta_2=0.999, epsilon=1e-8, amsgrad=False)
        model.compile(optimizer=optimiseur, loss='categorical_crossentropy', metrics=['accuracy'])
    elif optimiseur_entrainement == "SGD":
        optimiseur = SGD(lr=pas_entrainement, momentum=0.0, decay=0.0, nesterov=False)
        model.compile(optimizer=optimiseur, loss='categorical_crossentropy', metrics=['accuracy'])
    else:
        optimiseur = Adam(lr=pas_entrainement, beta_1=0.9, beta_2=0.999, epsilon=1e-8)
        model.compile(optimizer=optimiseur, loss='categorical_crossentropy', metrics=['accuracy'])
    #model.compile(optimizer=Adam(lr=pas_entrainement, beta_1=0.9, beta_2=0.999, epsilon=1e-8), loss='categorical_crossentropy', metrics=['accuracy'])







pythonpath = ""


