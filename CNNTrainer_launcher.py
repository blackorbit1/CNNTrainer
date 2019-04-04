#!C:/Users/EnzoGamer/AppData/Local/conda/conda/envs/tf_gpu/python.exe
path = "C:/Users/EnzoGamer/AppData/Local/conda/conda/envs/tf_gpu/python.exe"
"""
Auteur : DUTRA Enzo (14/2/2019)
"""

import subprocess
import traceback
import trace 
import threading

dependencies = ["xml", "lxml", "lxml.etree", "tkinter", "ntpath", "Augmentor", "torchvision"]

def import_or_install(package):
    try:
        __import__(package)
    except ImportError:
        print("Certains modules ne sont pas installés, installation ...")
        try:
            subprocess.check_call(["python3", '-m', 'pip', 'install', package]) # install pkg
            pip.main(['install', package])
        except:
            try:
                subprocess.check_call(["python3", '-m', 'pip', 'install', package]) # install pkg
                pip.main(['install', package])
            except:
                print("Les modules manquants n'ont pas pu etre installés, veuillez vous assurer que tous les modules suivants sont bien présents:")
                print(dependencies)
    except ModuleNotFoundError:
        print("Certains modules ne sont pas installés, installation ...")
        try:
            subprocess.check_call(["python3", '-m', 'pip', 'install', package]) # install pkg
            pip.main(['install', package])
        except:
            try:
                subprocess.check_call(["python3", '-m', 'pip', 'install', package]) # install pkg
                pip.main(['install', package])
            except:
                print("Les modules manquants n'ont pas pu etre installés, veuillez vous assurer que tous les modules suivants sont bien présents:")
                print(dependencies)


for package in dependencies:
    import_or_install(package)

from tkinter import *
from tkinter.filedialog import *
from tkinter import ttk as tkk
from tkinter.messagebox import showerror, showinfo
import tkinter as tk
from threading import Thread
import xml.etree.ElementTree as ET
from lxml import etree
import ntpath
import shutil
import Augmentor
import torchvision
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

def isfloat(value):
  try:
    float(value)
    return True
  except ValueError:
    return False

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

class Logger(object):
    def __init__(self, terminal):
        #self.printer = sys.stdout
        self.terminal = terminal
        self.train_progress = None
        #self.log = open("logfile.log", "a")    
    def write(self, message):
        #self.printer.write(message)
        self.terminal.insert(tk.END, message)
        self.terminal.see(tk.END)

        if self.train_progress is not None:
            if "Lancement de l'entrainement ..." in message:
                self.train_progress["value"] = 5
            if "Epoch " in message:
                avancement = message[6:]
                #print(avancement)
                actuel, final = avancement.split("/")
                #print("actuel : " + actuel + " / final : " + final)
                final_amount = int(actuel) * 100 / int(final)
                #print(int(final_amount))
                self.train_progress["value"] = final_amount
    def set_training_bar(self, training_bar):
        self.train_progress = training_bar
    def flush(self):
        #this flush method is needed for python 3 compatibility.
        #this handles the flush command by doing nothing.
        #you might want to specify some extra behavior here.
        pass    


class thread_with_trace(threading.Thread): 
  def __init__(self, *args, **keywords): 
    threading.Thread.__init__(self, *args, **keywords) 
    self.killed = False
  
  def start(self): 
    self.__run_backup = self.run 
    self.run = self.__run       
    threading.Thread.start(self) 
  
  def __run(self): 
    sys.settrace(self.globaltrace) 
    self.__run_backup() 
    self.run = self.__run_backup 
  
  def globaltrace(self, frame, event, arg): 
    if event == 'call': 
      return self.localtrace 
    else: 
      return None
  
  def localtrace(self, frame, event, arg): 
    if self.killed: 
      if event == 'line': 
        raise SystemExit() 
    return self.localtrace 
  
  def kill(self): 
    self.killed = True

# Idée de fonction du logiciel pour augmenter un dossier d'exemples
def run_dataset_configuration(data_dir, nb_images_augmentation, liste_options_augmentation):
    pass


def run_training(bouton_lancer_entrainement, nb_layers_to_freeze, reprise, reprise_poids, finetuning_partiel, dir_modele, nb_classes, dir_train, dir_train_nb_fic, dir_validation, dir_validation_nb_fic, preentrainement, nb_epoch_preentrainement, pas_preentrainement, nb_epoch_entrainement, pas_entrainement, batch_size, type_modele, optimiseur_preentrainement, optimiseur_entrainement):
    bouton_lancer_entrainement.config(state=DISABLED)
    print("\n")
    print(os.getcwd())

    from time import time
    from keras.callbacks import TensorBoard


    import glob
    import argparse
    #import matplotlib.pyplot as plt
    from keras.applications.inception_v3 import InceptionV3, preprocess_input
    from keras.applications.inception_resnet_v2 import InceptionResNetV2
    from keras.applications.resnet50 import ResNet50
    from keras.applications.nasnet import NASNetMobile
    #from keras.applications.mobilenet_v2 import NASNetMobile
    
    from keras.models import Model
    from keras.layers import Dense, GlobalAveragePooling2D
    from keras.preprocessing.image import ImageDataGenerator
    from keras.optimizers import Adam, RMSprop
    
    from tensorflow.python.client import device_lib
    import tensorflow as tf
    from keras.backend.tensorflow_backend import set_session
    

    
    
    config = tf.ConfigProto(
        gpu_options = tf.GPUOptions(force_gpu_compatible = True)
        # device_count = {'GPU': 1}
    )
    config.gpu_options.allow_growth = True
    session = tf.Session(config=config)
    set_session(session)
    
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    session = tf.Session(config=config)
    
    
    print(device_lib.list_local_devices())
    #tf.Session(config=tf.ConfigProto(allow_growth=True))
    
    # confirm TensorFlow sees the GPU
    from tensorflow.python.client import device_lib
    assert 'GPU' in str(device_lib.list_local_devices())
    
    # confirm Keras sees the GPU
    from keras import backend
    assert len(backend.tensorflow_backend._get_available_gpus()) > 0
    import time
    time.sleep(1)
    
    
    print("\nChargement des packages terminés\n")
    
    #nb_classes, dir_train, dir_train_nb_fic, dir_validation, dir_validation_nb_fic, preentrainement, nb_epoch_preentrainement, pas_preentrainement, nb_epoch_entrainement, pas_entrainement, batch_size = recevoir_parametres_CNN()
    
    """
    a = argparse.ArgumentParser()
    a.add_argument("--reprise")
    a.add_argument("--path_modele")
    a.add_argument("--type_modele")
    a.add_argument("--nb_classes")
    a.add_argument("--dir_train")
    a.add_argument("--dir_train_nb_fic")
    a.add_argument("--dir_validation")
    a.add_argument("--dir_validation_nb_fic")
    a.add_argument("--preentrainement")
    a.add_argument("--nb_epoch_preentrainement")
    a.add_argument("--pas_preentrainement")
    a.add_argument("--optimiseur_preentrainement")
    a.add_argument("--nb_epoch_entrainement")
    a.add_argument("--pas_entrainement")
    a.add_argument("--optimiseur_entrainement")
    a.add_argument("--batch_size")
    
    
    args = a.parse_args()
    
    reprise = True if reprise == "true" else False
    dir_modele = args.path_modele
    nb_classes = int(args.nb_classes)
    dir_train = args.dir_train
    dir_train_nb_fic = int(args.dir_train_nb_fic)
    dir_validation = args.dir_validation
    dir_validation_nb_fic = int(args.dir_validation_nb_fic)
    preentrainement = True if args.preentrainement == "true" else False
    nb_epoch_preentrainement = int(args.nb_epoch_preentrainement)
    pas_preentrainement = float(args.pas_preentrainement)
    nb_epoch_entrainement = int(args.nb_epoch_entrainement)
    pas_entrainement = float(args.pas_entrainement)
    batch_size = int(args.batch_size)
    
    type_modele = args.type_modele
    optimiseur_preentrainement = args.optimiseur_preentrainement
    optimiseur_entrainement = args.optimiseur_entrainement
    """
    
    
    
    
    
    
    if not reprise:
        # nom automatique du modele généré
        dir_modele = "model_"
        nb_modele_dir = 1
        while os.path.isfile(dir_modele + str(nb_modele_dir) + ".h5"):
            nb_modele_dir += 1
        dir_modele = dir_modele + str(nb_modele_dir) + ".h5"
    
    """   
    # Si l'user avait demandé de continuer l'entrainement d'un CNN déjà existant
    if user_input == 'c':
        while not os.path.isfile(dir_modele):
            entree_user = input("Donnez le chemin vers le modele que vous voulez continuer à entrainer :\n> ")
            dir_modele = entree_user
    """

    print("\nLe directory du fichier contenant le CNN qui sera enregistré : " + dir_modele + "\n")
    
    
    print("\nLancement de l'entrainement ...\n")

    """
    p_train = Augmentor.Pipeline()

    p_train.rotate(probability=0.7, max_left_rotation=10, max_right_rotation=10)
    p_train.flip_top_bottom(0.5)
    p_train.random_distortion(probability=1, grid_width=4, grid_height=4, magnitude=16)
    p_train.flip_left_right(probability=0.5)
    p_train.flip_top_bottom(probability=0.5)
    p_train.crop_random(probability=0.05, percentage_area=0.5)
    p_train.rotate90(probability=0.1)
    p_train.rotate270(probability=0.1)
    p_train.random_contrast(probability=0.1, min_factor=0.5, max_factor =1.5)
    p_train.random_brightness(probability=0.1, min_factor=0.5, max_factor =1.5)
    p_train.gaussian_distortion(probability=0.1, grid_width=16, grid_height=16, magnitude=2, corner="bell", method="in", mex=1.1,
                              mey=1.1, sdx=1.1, sdy=1.1)
    p_train.random_erasing(probability=0.05, rectangle_area=0.5)
    p_train.skew(probability=0.1, magnitude=0.2)
    p_train.greyscale(probability=0.01)
    p_train.histogram_equalisation(probability=0.1)
    """
    """
    if(os.path.exists(dir_train + "/output/")):
        shutil.rmtree(dir_train + "/output/")
    """
    
    #p_train.status()

    #g_train = p_train.keras_generator(batch_size = batch_size, image_data_format='channels_first')
    """g_train = p_train.keras_generator(batch_size = batch_size)"""
    """
    images, labels = next(g_train)
    print(labels[0])
    plt.imshow(images[0].reshape(28, 28), cmap="Greys");
    """
    """


    train_datagen =  ImageDataGenerator(p_train.keras_preprocess_func())
    train_generator = train_datagen.flow_from_directory(
        dir_train,
        target_size=(IM_WIDTH, IM_HEIGHT),
        batch_size=batch_size
    )


    
    p_validation = Augmentor.Pipeline()

    p_validation.rotate(probability=0.7, max_left_rotation=10, max_right_rotation=10)
    p_validation.flip_top_bottom(0.5)
    p_validation.random_distortion(probability=1, grid_width=4, grid_height=4, magnitude=16)
    p_validation.flip_left_right(probability=0.5)
    p_validation.flip_top_bottom(probability=0.5)
    p_validation.crop_random(probability=0.05, percentage_area=0.5)
    p_validation.rotate90(probability=0.1)
    p_validation.rotate270(probability=0.1)
    p_validation.random_contrast(probability=0.1, min_factor=0.5, max_factor =1.5)
    p_validation.random_brightness(probability=0.1, min_factor=0.5, max_factor =1.5)
    p_validation.gaussian_distortion(probability=0.1, grid_width=16, grid_height=16, magnitude=2, corner="bell", method="in", mex=1.1,
                              mey=1.1, sdx=1.1, sdy=1.1)
    p_validation.random_erasing(probability=0.05, rectangle_area=0.5)
    p_validation.skew(probability=0.1, magnitude=0.2)
    p_validation.greyscale(probability=0.01)
    p_validation.histogram_equalisation(probability=0.1)
    """
    

    """
    if(os.path.exists(dir_validation + "/output/")):
        shutil.rmtree(dir_validation + "/output/")
    """

    """
    #p_validation.status()

    
    #g_validation = p_validation.keras_generator(batch_size = batch_size, image_data_format='channels_first')
    g_validation = p_validation.keras_generator(batch_size = batch_size)

   
    test_datagen =  ImageDataGenerator(p_validation.keras_preprocess_func())
    validation_generator = test_datagen.flow_from_directory(
        dir_validation,
        target_size=(IM_WIDTH, IM_HEIGHT),
        batch_size=batch_size
    )
    """

    if(type_modele == "nasnetmobile"):
        IM_WIDTH, IM_HEIGHT = 224, 224
    else:
        IM_WIDTH, IM_HEIGHT = 299, 299 # fixed size for InceptionV3


   
    # data prep
    train_datagen =  ImageDataGenerator(
        preprocessing_function=preprocess_input,
        rotation_range=30,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True
    )
    
    train_generator = train_datagen.flow_from_directory(
        dir_train,
        target_size=(IM_WIDTH, IM_HEIGHT),
        batch_size=batch_size,
    )
   

    test_datagen = ImageDataGenerator(
        preprocessing_function=preprocess_input,
        rotation_range=30,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True
    )
    
    validation_generator = test_datagen.flow_from_directory(
        dir_validation,
        target_size=(IM_WIDTH, IM_HEIGHT),
        batch_size=batch_size,
    )
    
    
    
    if reprise:
        if reprise_poids:
            # setup model
            if(type_modele == "resnet"):
                base_model = ResNet50(weights='imagenet', include_top=False, input_shape=(299, 299, 3))
            elif(type_modele == "inceptionresnetv2"):
                base_model = InceptionResNetV2(weights='imagenet', include_top=False, input_shape=(299, 299, 3))
            elif(type_modele == "nasnetmobile"):
                base_model = NASNetMobile(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
            else:
                base_model = InceptionV3(weights='imagenet', include_top=False, input_shape=(299, 299, 3)) #include_top=False excludes final FC layer
            model = add_new_last_layer(base_model, nb_classes)
            model.load_weights(dir_modele)
        else:
            from keras.models import load_model
            print("\nchargement du modele ...")
            model = load_model(dir_modele)
            print("chargement du modele terminé\n")
            if finetuning_partiel:
                from keras.layers.convolutional import Convolution2D, MaxPooling2D
                from keras.layers.core import Dropout, Activation

                model.input

                #model.summary(line_length=150)
                model.layers.pop()
                model.layers.pop()
                model.layers.pop()
                #model.summary(line_length=150)
                
                # add a global spatial average pooling layer
                x = model.layers[-1].output
                x = GlobalAveragePooling2D()(x)
                # let's add a fully-connected layer
                x = Dense(1024, activation='relu')(x)
                # and a logistic layer -- let's say we have 200 classes
                predictions = Dense(nb_classes, activation='softmax')(x)
                
                model = Model(input=model.input, output=predictions)

    else:
        # setup model
        if(type_modele == "resnet"):
            base_model = ResNet50(weights='imagenet', include_top=False, input_shape=(299, 299, 3))
        elif(type_modele == "inceptionresnetv2"):
            base_model = InceptionResNetV2(weights='imagenet', include_top=False, input_shape=(299, 299, 3))
        elif(type_modele == "nasnetmobile"):
                base_model = NASNetMobile(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
        else:
            base_model = InceptionV3(weights='imagenet', include_top=False, input_shape=(299, 299, 3)) #include_top=False excludes final FC layer
        model = add_new_last_layer(base_model, nb_classes)

    tensorboard = TensorBoard(log_dir = "./logs", histogram_freq=0, write_graph=True, write_images=False)
    
    #model.summary() # Afficher le CNN en entier
    
    if preentrainement:
        # transfer learning
        setup_to_transfer_learn(model, base_model, pas_preentrainement, optimiseur_preentrainement)
    
        print("\n\n--- --- Lancement du pré-entrainement --- ---\n")
        history_tl = model.fit_generator(
            train_generator,
            nb_epoch=nb_epoch_preentrainement,
            workers=1,
            use_multiprocessing=False,
            steps_per_epoch=dir_train_nb_fic // batch_size,
            validation_steps=dir_validation_nb_fic // batch_size,
            samples_per_epoch=dir_train_nb_fic,
            validation_data=validation_generator,
            #nb_val_samples=dir_validation_nb_fic,
            #class_weight='auto'
        )
    
    
    # fine-tuning
    setup_to_finetune(model, pas_entrainement, optimiseur_entrainement, nb_layers_to_freeze)
    
    print("\n\n--- --- Lancement de l'entrainement --- ---\n")
    """
    print(dir_validation_nb_fic // batch_size)
    print(dir_validation_nb_fic)
    print(batch_size)
    print(args.dir_validation_nb_fic)
    print(args.batch_size)
    """
    #from sklearn.utils import class_weight
    #import numpy as np
    # train the model on the new data for a few epochs
    #class_weight = class_weight.compute_class_weight('balanced', np.unique(train_generator), train_generator)
    
    history_ft = model.fit_generator(
        train_generator,                                        # generateur de nouvelles image d'entrainement
        samples_per_epoch=dir_train_nb_fic,                     # nb de fichiers d'entrainement
        epochs=nb_epoch_entrainement,                            # nb de cycles d'entrainement
        workers=1,                                              # nb d'user travaillant dessus (laisser 1 si GPU)
        use_multiprocessing=False,                              # laisser False si GPU
        #steps_per_epoch=dir_train_nb_fic // batch_size,         # nb fic entrainement / taille tampon
        validation_steps=dir_validation_nb_fic // batch_size,   # nb fic validation / taille tampon
        validation_data=validation_generator,                   # generateur de nouvelles image de validation
        #nb_val_samples=dir_validation_nb_fic,                  # nb fichiers de validation (laisser en com)
        class_weight="auto",
        callbacks = [tensorboard],
        shuffle=True
        #verbose=2
    )
    
    print("""
        ╔══════════════════════════════════════════════════════╗
        ║          Entrainement terminé, à bientot !           ║
        ╚══════════════════════════════════════════════════════╝
    """)
    
    print("\n\nEnregistrement du fichier " + dir_modele + " ....")
    model.save(dir_modele)  # always save your weights after training or during training
    print("fichier enregistré !")
    bouton_lancer_entrainement.config(state=NORMAL)



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


def setup_to_finetune(model, pas_entrainement, optimiseur_entrainement, nb_layers_to_freeze):
    from keras.optimizers import Adam, RMSprop, SGD
    """Freeze the bottom NB_IV3_LAYERS and retrain the remaining top layers.
    note: NB_IV3_LAYERS corresponds to the top 2 inception blocks in the inceptionv3 arch
    Args:
    model: keras model
    """
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




def leterminal(command, terminal, processing_bar, bouton_lancer_entrainement):

    original = sys.stdout
    sys.stdout = Logger(terminal)
    
    print("test de la fonction print")

    running = True
    bouton_lancer_entrainement.config(state=DISABLED)
    processing_bar["value"] = 1
    p = subprocess.Popen(command, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, universal_newlines=True) #, env={'LANGUAGE':'en_US.en', 'LC_ALL':'en_US.UTF-8'}
    p.poll()
    processing_bar["value"] = 3

    terminal.insert(tk.END, "> Lancement de l'execution du script d'entrainement")
    terminal.see(tk.END)

    while running:
        line = p.stdout.readline()
        """
        err = p.stderr.readline()
        terminal.insert(tk.END, err)
        terminal.see(tk.END)
        """
        if "Enregistrement du fichier" in line:
            running = False
        if "Lancement de l'entrainement ..." in line:
            processing_bar["value"] = 5
        if "Epoch " in line:
            avancement = line[6:]
            #print(avancement)
            actuel, final = avancement.split("/")
            #print("actuel : " + actuel + " / final : " + final)
            final_amount = int(actuel) * 100 / int(final)
            #print(int(final_amount))
            processing_bar["value"] = final_amount

        terminal.insert(tk.END, line)
        terminal.see(tk.END)
        if not line and p.poll is not None: break

    """
    while running:
        err = p.stderr.readline()
        terminal.insert(tk.END, err)
        terminal.see(tk.END)
        if not err and p.poll is not None: break
    """

    processing_bar["value"] = 0
    bouton_lancer_entrainement.config(state=NORMAL)
    terminal.insert(tk.END, "-")

    sys.stdout = original


def changer_xml(balise, attribut, valeur):
    tree = ET.parse("CNNTrainer\\base_de_donnees.xml")
    root = tree.getroot()
    for variable in root.iter(balise):
        variable.get(attribut)
        variable.set(attribut, valeur)
    tree.write("CNNTrainer\\base_de_donnees.xml")

def voir_xml(balise, attribut):
    tree = etree.parse("CNNTrainer\\base_de_donnees.xml")
    for balise in tree.xpath("/data_base/" + balise):
        return balise.get(attribut)


pythonpath = ""



class Interface(Frame):

    """Notre fenêtre principale.
    Tous les widgets sont stockés comme attributs de cette fenêtre."""

    def __init__(self, fenetre, **kwargs):
        fenetre.config(height=1000, width=400)
        Frame.__init__(self, fenetre, width=200, height=200, **kwargs)
        self.pack(fill=BOTH)

        self.padding = 10

        self.modelpath = ""
        self.trainpath = ""
        self.validationpath = ""
        self.pythonpath = ""
        self.nb_classes = 0
        self.nb_classes_train = 0
        self.nb_classes_validation = 0

        self.thread_entrainement = None

        temp = path
        if(os.path.isfile(temp)):
                self.pythonpath = temp

        # Création de nos widgets
        self.logo = PhotoImage(file="CNNTrainer\\top_logo.png")
        self.logocadre = Canvas(self, width=867, height=75)
        self.logocadre.create_image(0, 0, anchor=NW, image=self.logo)
        self.logocadre.place(x=200, y=200, anchor=NW)
        self.logocadre.pack()

        self.n = tkk.Notebook(self)
        self.f1 = Frame(self.n)   # first page, which would get widgets gridded into it
        self.f2 = Frame(self.n)   # second page
        self.n.add(self.f1, text='  Entrainement CNN  ')
        self.n.add(self.f2, text='  Test CNN  ')
        self.n.pack()


        ### --- Sortie du script --- ###

        self.text = Text(self.f1)
        self.text.config(font=("Courrier New", 8))
        self.text.pack(fill=Y, side=RIGHT)


        ### --- Reprise d'entrainement --- ###

        self.continuer_entraienment = LabelFrame(self.f1, text="Reprise d'entrainement")
        self.continuer_entraienment.pack(fill=X, side=TOP, expand="yes", padx=self.padding, pady=self.padding)

        self.continue_train = IntVar()
        self.case_continue_train = Checkbutton(self.continuer_entraienment,
                                               text="Continuer un entrainement",
                                               variable=self.continue_train,
                                               command=self.griser_bouton_continuer)
        self.case_continue_train.grid(row=0, columnspan=2)

        self.label_modele_repris = Label(self.continuer_entraienment, text="Aucun fichier ")
        self.label_modele_repris.grid(row=1)

        #self.filepath = askopenfilename(title="Ouvrir une image",filetypes=[('png files','.png'),('all files','.*')])
        #self.photo = PhotoImage(file=self.filepath)
        self.bouton_choose_model = Button(self.continuer_entraienment, text="Choisir un modèle", command=self.choose_model, state=DISABLED)
        self.bouton_choose_model.grid(row=1, column=1)

        self.only_weights = IntVar()
        self.case_only_weights = Checkbutton(self.continuer_entraienment,
                                               text="Ne contient que les poids",
                                               variable=self.only_weights,
                                               command=self.ne_contient_que_les_poids)
        self.case_only_weights.grid(row=2, columnspan=2)
        self.case_only_weights.config(state=DISABLED)

        self.valeur_finetuning_partiel = IntVar()
        self.case_finetuning_partiel = Checkbutton(self.continuer_entraienment,
                                               text="Faire un fine tuning partiel",
                                               variable=self.valeur_finetuning_partiel,
                                               command=self.faire_finetuning_partiel)
        self.case_finetuning_partiel.grid(row=3, columnspan=2)
        self.case_finetuning_partiel.config(state=DISABLED)


        ### --- Base --- ###

        self.base = LabelFrame(self.f1, text="Base")
        self.base.pack(fill=X, side=TOP, expand="yes", padx=self.padding, pady=self.padding)

        self.label_nb_epoch = Label(self.base, text="Type de modele: ")
        self.label_nb_epoch.grid(row=0)


        self.menu_deroulant_valeur_modeles = StringVar(value='Inception V3')
        self.menu_deroulant_modeles = tkk.Combobox(self.base, textvariable=self.menu_deroulant_valeur_modeles)
        self.menu_deroulant_modeles.grid(row=0, column=1)
        self.menu_deroulant_modeles.bind('>', self.on_value_change)
        self.liste_modeles = ["Inception V3",  "Inception-ResNet V2", "ResNet", "NASNetMobile"]
        self.menu_deroulant_modeles['values'] =  self.liste_modeles




        self.label_batch_size = Label(self.base, text="Taille batch: ")
        self.label_batch_size.grid(row=1)

        self.menu_deroulant_valeur_batch_size = StringVar(value='32')
        self.menu_deroulant_batch_size = tkk.Combobox(self.base, textvariable=self.menu_deroulant_valeur_batch_size)
        self.menu_deroulant_batch_size.grid(row=1, column=1)
        self.menu_deroulant_batch_size.bind('>', self.on_value_change)
        self.liste_batch_size = ["8", "16", "32", "64", "128"]
        self.menu_deroulant_batch_size['values'] =  self.liste_batch_size

        """
        self.liste_types_modeles = ('Inception V3', 'ResNet')
        self.texte_liste_types_modeles = StringVar()
        self.texte_liste_types_modeles.set(self.liste_types_modeles[0])
        self.menu_deroulant_modeles = OptionMenu(self.types_modele, self.texte_liste_types_modeles, *self.liste_types_modeles)
        self.menu_deroulant_modeles.pack(side="right")
        """

        self.label_layers_to_freeze = Label(self.base, text="Nb layers to freeze: ")
        self.label_layers_to_freeze.grid(row=2)

        self.menu_deroulant_valeur_layers_to_freeze = StringVar(value='249')
        self.menu_deroulant_layers_to_freeze = tkk.Combobox(self.base, textvariable=self.menu_deroulant_valeur_layers_to_freeze)
        self.menu_deroulant_layers_to_freeze.grid(row=2, column=1)
        self.menu_deroulant_layers_to_freeze.bind('>', self.on_value_change)
        self.liste_layers_to_freeze = ["10", "50", "100", "150", "249"]
        self.menu_deroulant_layers_to_freeze['values'] =  self.liste_layers_to_freeze

        


        ### --- Pre-entrainement --- ###

        self.pre_entrainement = LabelFrame(self.f1, text="Pre-entrainement")
        self.pre_entrainement.pack(fill=X, side=TOP, expand="yes", padx=self.padding, pady=self.padding)

        self.active_pre_entrainement = IntVar()
        self.case_pre_entrainement = Checkbutton(self.pre_entrainement,
                                               text="Faire un pré-entrainement",
                                               variable=self.active_pre_entrainement,
                                               command=self.griser_interface_preentrainement)
        self.case_pre_entrainement.grid(row=0, columnspan=2)
        self.case_pre_entrainement.select()


        self.label_pas_preentrainement = Label(self.pre_entrainement, text="Taille pas: ").grid(row=1)

        self.menu_deroulant_valeur_pas_preentrainement = StringVar(value='0.001')
        self.menu_deroulant_pas_preentrainement = tkk.Combobox(self.pre_entrainement, textvariable=self.menu_deroulant_valeur_pas_preentrainement)
        self.menu_deroulant_pas_preentrainement.grid(row=1, column=1)
        self.menu_deroulant_pas_preentrainement.bind('>', self.on_value_change)
        self.liste_pas_preentrainement = ["0.01", "0.001", "0.0001", "0.00001"]
        self.menu_deroulant_pas_preentrainement['values'] =  self.liste_pas_preentrainement
        """
        self.liste_pas = ('0.001', '0.0001')
        self.texte_liste_pas = StringVar()
        self.texte_liste_pas.set(self.liste_pas[0])

        self.menu_deroulant_pas_preentrainement = OptionMenu(self.pre_entrainement,
                                                               self.texte_liste_pas,
                                                               *self.liste_pas)
        self.menu_deroulant_pas_preentrainement.pack(side="right")
        """

        self.label_epoch_preentrainement = Label(self.pre_entrainement, text="Nb epoch: ")
        self.label_epoch_preentrainement.grid(row=2)

        self.menu_deroulant_valeur_epoch_preentrainement = StringVar(value='20')
        self.menu_deroulant_epoch_preentrainement = tkk.Combobox(self.pre_entrainement, textvariable=self.menu_deroulant_valeur_epoch_preentrainement)
        self.menu_deroulant_epoch_preentrainement.grid(row=2, column=1)
        self.menu_deroulant_epoch_preentrainement.bind('>', self.on_value_change)
        self.liste_epoch_preentrainement = ["10", "20", "50", "100", "250", "500", "1000"]
        self.menu_deroulant_epoch_preentrainement['values'] =  self.liste_epoch_preentrainement



        self.label_optimiseur_preentrainement = Label(self.pre_entrainement, text="Optimiseur: ")
        self.label_optimiseur_preentrainement.grid(row=3)

        self.menu_deroulant_valeur_optimiseur_preentrainement = StringVar(value='RMSprop')
        self.menu_deroulant_optimiseur_preentrainement = tkk.Combobox(self.pre_entrainement, textvariable=self.menu_deroulant_valeur_optimiseur_preentrainement)
        self.menu_deroulant_optimiseur_preentrainement.grid(row=3, column=1)
        self.menu_deroulant_optimiseur_preentrainement.bind('>', self.on_value_change)
        self.liste_optimiseur_preentrainement = ["RMSprop", "Adam", "AMSGrad", "SGD"]
        self.menu_deroulant_optimiseur_preentrainement['values'] =  self.liste_optimiseur_preentrainement



        """
        self.label_epoch_preentrainement = Label(self.pre_entrainement, text="Taille pas: ")
        self.label_epoch_preentrainement.pack(side="left")

        self.liste_epoch = ('0.001', '0.0001')
        self.texte_liste_epoch = StringVar()
        self.texte_liste_epoch.set(self.liste_epoch[0])

        self.menu_deroulant_epoch_preentrainement = OptionMenu(self.pre_entrainement,
                                                               self.texte_liste_epoch,
                                                               *self.liste_epoch)
        self.menu_deroulant_epoch_preentrainement.pack(side="right")
        """




        ### --- Entrainement --- ###

        self.entrainement = LabelFrame(self.f1, text="Entrainement")
        self.entrainement.pack(fill=X, side=TOP, expand="yes", padx=self.padding, pady=self.padding)


        self.label_pas_entrainement = Label(self.entrainement, text="Taille pas: ").grid(row=1)

        self.menu_deroulant_valeur_pas_entrainement = StringVar(value='0.001')
        self.menu_deroulant_pas_entrainement = tkk.Combobox(self.entrainement, textvariable=self.menu_deroulant_valeur_pas_entrainement)
        self.menu_deroulant_pas_entrainement.grid(row=1, column=1)
        self.menu_deroulant_pas_entrainement.bind('>', self.on_value_change)
        self.liste_pas_entrainement = ["0.01", "0.001", "0.0001", "0.00001"]
        self.menu_deroulant_pas_entrainement['values'] =  self.liste_pas_entrainement



        self.label_epoch_entrainement = Label(self.entrainement, text="Nb epoch: ")
        self.label_epoch_entrainement.grid(row=2)

        self.menu_deroulant_valeur_epoch_entrainement = StringVar(value='50')
        self.menu_deroulant_epoch_entrainement = tkk.Combobox(self.entrainement, textvariable=self.menu_deroulant_valeur_epoch_entrainement)
        self.menu_deroulant_epoch_entrainement.grid(row=2, column=1)
        self.menu_deroulant_epoch_entrainement.bind('>', self.on_value_change)
        self.liste_epoch_entrainement = ["10", "20", "50", "100", "250", "500", "1000"]
        self.menu_deroulant_epoch_entrainement['values'] =  self.liste_epoch_entrainement



        self.label_optimiseur_entrainement = Label(self.entrainement, text="Optimiseur: ")
        self.label_optimiseur_entrainement.grid(row=3)

        self.menu_deroulant_valeur_optimiseur_entrainement = StringVar(value='Adam')
        self.menu_deroulant_optimiseur_entrainement = tkk.Combobox(self.entrainement, textvariable=self.menu_deroulant_valeur_optimiseur_entrainement)
        self.menu_deroulant_optimiseur_entrainement.grid(row=3, column=1)
        self.menu_deroulant_optimiseur_entrainement.bind('>', self.on_value_change)
        self.liste_optimiseur_entrainement = ["RMSprop", "Adam", "AMSGrad", "SGD"]
        self.menu_deroulant_optimiseur_entrainement['values'] =  self.liste_optimiseur_entrainement


        ### --- Dataset --- ###

        self.dataset = LabelFrame(self.f1, text="Dataset")
        self.dataset.pack(fill=X, side=TOP, expand="yes", padx=self.padding, pady=self.padding)

        self.label_dataset_train = Label(self.dataset, text="0 photo, 0 classe")
        self.label_dataset_train.grid(row=1)

        self.bouton_choose_dataset_train = Button(self.dataset, text="Dossier d'entrainement", command=self.choose_dataset_train)
        self.bouton_choose_dataset_train.grid(row=1, column=1)


        self.label_dataset_validation = Label(self.dataset, text="0 photo, 0 classe")
        self.label_dataset_validation.grid(row=2)

        self.bouton_choose_dataset_validation = Button(self.dataset, text="Dossier de validation", command=self.choose_dataset_validation)
        self.bouton_choose_dataset_validation.grid(row=2, column=1)


        ### --- Lancer l'entrainement --- ###

        self.cadre_lancer_entrainement = Frame(self.f1, borderwidth=20, relief=FLAT)
        self.cadre_lancer_entrainement.pack(fill=X, side=TOP, expand="yes", padx=self.padding, pady=self.padding)

        self.bouton_lancer_entrainement = Button(self.cadre_lancer_entrainement, text="Lancer l'entrainement !", fg="green",command=self.verification_et_lancement)
        self.bouton_lancer_entrainement.pack(side="left")

        self.bouton_stopper_entrainement = Button(self.cadre_lancer_entrainement, text="Stopper l'entrainement", fg="red", command=self.stopper_entrainement)
        self.bouton_stopper_entrainement.pack(side="right")

        """
        import tkinter.font
        for name in sorted(tkinter.font.families()):
            print(name)
        """

        ### --- Bar de chargement --- ###

        # Prepare the type of Progress bar needed.
        # Look out for determinate mode and pick the suitable one
        # Other formats of display can be suitably explored
        self.processing_bar = tkk.Progressbar(self.f1, orient='horizontal', length=300, value=0, maximum=100)

        # Place the bar at the centre of the window
        self.processing_bar.place(relx=0.5, rely=0.5, anchor=CENTER)
        self.processing_bar.pack(fill=X, side=BOTTOM, expand="yes")


        #command = "C:\\Users\\EnzoGamer\\AppData\\Local\\conda\\conda\\envs\\tf_gpu\\python.exe \"C:\\Users\\EnzoGamer\\Desktop\\PROJET IA\\cnn trainer\\CNNTrainer\\script.py\""

        #t = Thread(target = lambda: leterminal(command, self.text))
        #t.start()







        ### ### ### === === TEST D'UN MODELE === === ### ### ###






        self.trainedmodelpath = ""
        self.liste_classes = []
        self.nb_classes = 0
        self.cnn_a_tester = None


        ### --- Choix du modele a tester --- ###

        self.test_model = LabelFrame(self.f2, text="Modèle")
        self.test_model.grid(row=0, padx=self.padding, pady=self.padding)

        self.label_test_model = Label(self.test_model, text="Aucun fichier ")
        self.label_test_model.grid(row=1)

        #self.filepath = askopenfilename(title="Ouvrir une image",filetypes=[('png files','.png'),('all files','.*')])
        #self.photo = PhotoImage(file=self.filepath)
        self.bouton_choose_test_model = Button(self.test_model, text="Choisir un modèle", command=self.choose_test_model)
        self.bouton_choose_test_model.grid(row=1, column=1)

        
        ### --- Indication des différentes classes --- ###


        self.test_classes = LabelFrame(self.f2, text="Classes")
        self.test_classes.grid(row=1, padx=self.padding, pady=self.padding)

        self.label_nb_classes_test = Label(self.test_classes, text="0 classe trouvée")
        self.label_nb_classes_test.pack()

        self.texte_classes = Text(self.test_classes)
        self.texte_classes.config(font=("Source Code Pro", 8), width=50)
        self.texte_classes.pack()

        self.cadre_actualiser_classes = Frame(self.test_classes, borderwidth=20, relief=FLAT)
        self.cadre_actualiser_classes.pack()

        self.bouton_lancer_entrainement = Button(self.cadre_actualiser_classes, text="Actualiser", command=self.actualiser_classes_test)
        self.bouton_lancer_entrainement.pack(side="bottom")


        ### --- Test de l'image --- ###

        self.test_image = LabelFrame(self.f2, text="Test")
        self.test_image.grid(row=0, column=1, rowspan=20, padx=self.padding, pady=self.padding)

        self.test_image_cadre = Canvas(self.test_image, width=500, height=500)
        #self.test_image_cadre.create_image(0, 0, anchor=NW)
        self.test_image_cadre.grid(row=0, columnspan=2)

        self.label_test_image = Label(self.test_image, text="Image : Aucun fichier ")
        self.label_test_image.grid(row=1)

        #self.filepath = askopenfilename(title="Ouvrir une image",filetypes=[('png files','.png'),('all files','.*')])
        #self.photo = PhotoImage(file=self.filepath)
        self.bouton_choose_test_image = Button(self.test_image, text="Choisir une image", command=self.choose_test_image)
        self.bouton_choose_test_image.grid(row=1, column=1)


        self.label_test_reponse = Label(self.test_image, text="...")
        self.label_test_reponse.grid(row=2)

        self.label_test_suretee = Label(self.test_image, text="0 %")
        self.label_test_suretee.grid(row=3)







        self.label_pythonpath = Label(self, text="Chemin vers l'interpreteur python: " + self.pythonpath)
        self.label_pythonpath.pack(side="left")

        self.bouton_cliquer = Button(self, text="Changer d'interpreteur python", fg="red", command=self.change_pyhton_path)
        self.bouton_cliquer.pack(side="right")

    def verification_et_lancement(self): ### --- --- On ne peut pas juste faire un include sinon on a pas la sortie du terminal --- --- ###
        reprise = False
        path_modele = ""
        type_modele = "Inception V3"
        nb_classes = 0
        dir_train = ""
        dir_train_nb_fic = 0
        dir_validation = ""
        dir_validation_nb_fic = 0
        preentrainement = False
        nb_epoch_preentrainement = 0
        pas_preentrainement = 0
        optimiseur_preentrainement = ""
        nb_epoch_entrainement = 0
        pas_entrainement = 0
        optimiseur_entrainement = ""
        batch_size = 0
        reprise_poids = True
        finetuning_partiel = False
        nb_layers_to_freeze = 249
        
        erronne = False
        erreurs = []

        try:


            if(self.nb_classes_train == 0 and self.nb_classes_validation == 0):
                erronne = True
                erreurs.append("Aucune classe n'est fournie")
            elif(self.nb_classes_train != self.nb_classes_validation):
                erronne = True
                erreurs.append("Le nombre de classes des dossiers de validation et d'entrainement sont différents")
            else:
                nb_classes = self.nb_classes_train
                dir_train = self.trainpath
                for root, dirs, files in os.walk(self.trainpath):
                    dir_train_nb_fic += len(files)
                #dir_train_nb_fic = int(len(next(os.walk(self.trainpath))[1]))
                dir_validation = self.validationpath
                for root, dirs, files in os.walk(self.validationpath):
                    dir_validation_nb_fic += len(files)
                #dir_validation_nb_fic = int(len(next(os.walk(self.validationpath))[1]))

            try:
                batch_size = int(self.menu_deroulant_valeur_batch_size.get())
            except:
                erronne = True
                erreurs.append("La valeur donnée pour la taille du batch (tampon) n'est pas un entier valide")

            if(self.continue_train.get() == 1):
                reprise = True
                path_modele = self.modelpath

            if self.menu_deroulant_valeur_modeles.get() in ["Inception V3", "ResNet", "Inception-ResNet V2", "NASNetMobile"]:
                if(self.menu_deroulant_valeur_modeles.get() == "Inception V3"):
                    type_modele = "inceptionv3"
                elif(self.menu_deroulant_valeur_modeles.get() == "ResNet"):
                    type_modele = "resnet"
                elif(self.menu_deroulant_valeur_modeles.get() == "Inception-ResNet V2"):
                    type_modele = "inceptionresnetv2"
                elif(self.menu_deroulant_valeur_modeles.get() == "NASNetMobile"):
                    type_modele = "nasnetmobile"
            else:
                erronne = True
                erreurs.append("Le modele demandé n'est pas supporté, veuillez entrer un modele parmis ceux proposés (il est possible que vous ayez fait une erreur de frappe)")

            if self.active_pre_entrainement.get() == 1:
                preentrainement = True
                try:
                    pas_preentrainement = float(self.menu_deroulant_valeur_pas_preentrainement.get())
                except:
                    erronne = True
                    erreurs.append("La valeur donnée pour la taille de pas du préentrainement n'est pas un reel valide")
                try:
                    nb_epoch_preentrainement = int(self.menu_deroulant_valeur_epoch_preentrainement.get())
                except:
                    erronne = True
                    erreurs.append("La valeur donnée pour le nb d'epoch du préentrainement n'est pas un entier valide")
                if self.menu_deroulant_valeur_optimiseur_preentrainement.get() in ["RMSprop", "Adam", "AMSGrad", "SGD"]:
                    optimiseur_preentrainement = self.menu_deroulant_valeur_optimiseur_preentrainement.get()
                else:
                    erronne = True
                    erreurs.append("L'optimiseur demandé n'est pas supporté, veuillez entrer un optimiseur parmis ceux proposés (il est possible que vous ayez fait une erreur de frappe)")


            try:
                pas_entrainement = float(self.menu_deroulant_valeur_pas_entrainement.get())
            except:
                erronne = True
                erreurs.append("La valeur donnée pour la taille de pas de l'entrainement n'est pas un reel valide")
            try:
                nb_epoch_entrainement = int(self.menu_deroulant_valeur_epoch_entrainement.get())
            except:
                erronne = True
                erreurs.append("La valeur donnée pour le nb d'epoch du entrainement n'est pas un entier valide")
            if self.menu_deroulant_valeur_optimiseur_entrainement.get() in ["RMSprop", "Adam", "AMSGrad", "SGD"]:
                optimiseur_entrainement = self.menu_deroulant_valeur_optimiseur_entrainement.get()
            else:
                erronne = True
                erreurs.append("L'optimiseur demandé n'est pas supporté, veuillez entrer un optimiseur parmis ceux proposés (il est possible que vous ayez fait une erreur de frappe)")
            
            if int(self.menu_deroulant_valeur_layers_to_freeze.get()) >= 0:
                nb_layers_to_freeze = int(self.menu_deroulant_valeur_layers_to_freeze.get())
            else:
                erronne = True
                erreurs.append("La valeur du nombre de couches à rendre non-entrainables est erroné")

            if self.only_weights.get() == 1:
                reprise_poids = True
            else:
                reprise_poids = False

            if self.valeur_finetuning_partiel.get() == 1:
                finetuning_partiel = True
            else:
                finetuning_partiel = False


            if erreurs:
                texte = ""
                for erreur in erreurs:
                    texte += " - " + erreur + "\n"
                showerror("Erreur", texte)

        except Exception as e:
            showerror("Erreur", "Erreur lors de l'enregistrement des parametres")
            print(e)
            traceback.print_stack()

        if not erronne:
            #run_training(reprise, path_modele, nb_classes, dir_train, dir_train_nb_fic, dir_validation, dir_validation_nb_fic, preentrainement, nb_epoch_preentrainement, pas_preentrainement, nb_epoch_entrainement, pas_entrainement, batch_size, type_modele, optimiseur_preentrainement, optimiseur_entrainement)
            """
            commande = ''

            commande += self.pythonpath
            commande += ' "' + os.getcwd() + '\\CNNTrainer\\script.py"'

            commande += ' --reprise "' + ("true" if reprise else "false") + '"'
            commande += ' --path_modele "' + path_modele + '"'
            commande += ' --type_modele "' + type_modele + '"'
            commande += ' --nb_classes  "' + str(nb_classes) + '"'
            commande += ' --dir_train "' + dir_train + '"'
            commande += ' --dir_train_nb_fic "' + str(dir_train_nb_fic) + '"'
            commande += ' --dir_validation "' + dir_validation + '"'
            commande += ' --dir_validation_nb_fic "' + str(dir_validation_nb_fic) + '"'
            commande += ' --preentrainement "' + ("true" if preentrainement else "false") + '"'
            commande += ' --nb_epoch_preentrainement "' + str(nb_epoch_preentrainement) + '"'
            commande += ' --pas_preentrainement "' + str(pas_preentrainement) + '"'
            commande += ' --optimiseur_preentrainement "' + optimiseur_preentrainement + '"'
            commande += ' --nb_epoch_entrainement "' + str(nb_epoch_entrainement) + '"'
            commande += ' --pas_entrainement "' + str(pas_entrainement) + '"'
            commande += ' --optimiseur_entrainement "' + optimiseur_entrainement + '"'
            commande += ' --batch_size "' + str(batch_size) + '"'

            print(commande)
            run_training(self.bouton_lancer_entrainement, reprise, path_modele, nb_classes, dir_train, dir_train_nb_fic, dir_validation, dir_validation_nb_fic, preentrainement, nb_epoch_preentrainement, pas_preentrainement, nb_epoch_entrainement, pas_entrainement, batch_size, type_modele, optimiseur_preentrainement, optimiseur_entrainement)
            """

            sys.stdout = Logger(self.text)
            sys.stderr = Logger(self.text)
            sys.stdout.set_training_bar(self.processing_bar)
            sys.stderr.set_training_bar(self.processing_bar)

            self.thread_entrainement = thread_with_trace(target = lambda: run_training(self.bouton_lancer_entrainement, nb_layers_to_freeze, reprise, reprise_poids, finetuning_partiel, path_modele, nb_classes, dir_train, dir_train_nb_fic, dir_validation, dir_validation_nb_fic, preentrainement, nb_epoch_preentrainement, pas_preentrainement, nb_epoch_entrainement, pas_entrainement, batch_size, type_modele, optimiseur_preentrainement, optimiseur_entrainement))
            self.thread_entrainement.start()


    def stopper_entrainement(self):
        self.thread_entrainement.kill()
        self.thread_entrainement.join()

        if not self.thread_entrainement.isAlive():
            print("=== === Entrainement stoppé === ===")



    def choose_dataset_train(self):
        temp = askdirectory(title="Dossier contenant les images à utiliser lors de l'entrainement")
        if(os.path.exists(temp) and int(len(next(os.walk(temp))[1])) > 0):
            dir_train_nb_fic = 0
            for root, dirs, files in os.walk(temp):
                dir_train_nb_fic += len(files)
            #print(dir_train_nb_fic)

            if(dir_train_nb_fic > 0):
                self.trainpath = temp
                self.nb_classes_train = int(len(next(os.walk(temp))[1]))
                self.label_dataset_train["text"] = str(dir_train_nb_fic) + " photos, " + str(self.nb_classes_train) + " classes"
            else:
                showerror("Aucune photo trouvée", "le dossier d'entrainement que vous avez donné semble contenir des classes mais ne contient aucune photo !")
        elif(int(len(next(os.walk(temp))[1])) == 0):
            showerror("Aucune classe trouvée", "le dossier d'entrainement que vous avez donné ne contient aucune classe !")
            self.label_dataset_train["text"] = "0 photo"
        else:
            showerror("Chemin invalide", "le chemin que vous avez donné n'est pas valide !")

    def choose_dataset_validation(self):
        temp = askdirectory(title="Dossier contenant les images à utiliser pour vérifier la performance du CCN pendant l'entrainement")
        if(os.path.exists(temp) and int(len(next(os.walk(temp))[1])) > 0):
            dir_validation_nb_fic = 0
            for root, dirs, files in os.walk(temp):
                dir_validation_nb_fic += len(files)
            #print(dir_validation_nb_fic)

            if(dir_validation_nb_fic > 0):
                self.validationpath = temp
                self.nb_classes_validation = int(len(next(os.walk(temp))[1]))
                self.label_dataset_validation["text"] = str(dir_validation_nb_fic) + " photos, " + str(self.nb_classes_validation) + " classes"
            else:
                showerror("Aucune photo trouvée", "le dossier de validation que vous avez donné semble contenir des classes mais ne contient aucune photo !")
        elif(int(len(next(os.walk(temp))[1])) == 0):
            showerror("Aucune classe trouvée", "le dossier de validation que vous avez donné ne contient aucune classe !")
            self.label_dataset_validation["text"] = "0 photo"
        else:
            showerror("Chemin invalide", "le chemin que vous avez donné n'est pas valide !")
    def on_value_change(self):
        print("méthode on_value_change() utilisée")
    def griser_interface_preentrainement(self):
        if(self.active_pre_entrainement.get() == 1):
            self.menu_deroulant_pas_preentrainement.config(state=NORMAL)
            self.menu_deroulant_epoch_preentrainement.config(state=NORMAL)
            self.menu_deroulant_optimiseur_preentrainement.config(state=NORMAL)
        else:
            self.menu_deroulant_pas_preentrainement.config(state=DISABLED)
            self.menu_deroulant_epoch_preentrainement.config(state=DISABLED)
            self.menu_deroulant_optimiseur_preentrainement.config(state=DISABLED)
    def griser_bouton_continuer(self):
        if(self.continue_train.get() == 1):
            self.bouton_choose_model.config(state=NORMAL)
            self.case_only_weights.config(state=NORMAL)
            self.case_finetuning_partiel.config(state=NORMAL)
            if self.only_weights.get() == 1 :
                self.menu_deroulant_modeles.config(state=NORMAL)
            else:
                self.menu_deroulant_modeles.config(state=DISABLED)
        else:
            self.bouton_choose_model.config(state=DISABLED)
            self.case_only_weights.config(state=DISABLED)
            self.case_finetuning_partiel.config(state=DISABLED)
            self.menu_deroulant_modeles.config(state=NORMAL)
    def ne_contient_que_les_poids(self):
        if(self.only_weights.get() == 1):
            self.menu_deroulant_modeles.config(state=NORMAL)
        else:
            self.menu_deroulant_modeles.config(state=DISABLED)
    def faire_finetuning_partiel(self):
        pass
    def change_pyhton_path(self):
        temp = askopenfilename(title="Indiquer l'interpreteur python à utiliser",filetypes=[('python','.*'),('all files','.*')])
        if(os.path.isfile(temp)):
            self.pythonpath = temp
            change_path(temp)
            #changer_xml(balise="pyhton_path", attribut="path", valeur=temp)
        else:
            print("le chemin que vous venez de donner n'est pas valide")
        print("affichage de pythonpath :")
        print(self.pythonpath)
        print("modification du label de texte ..")
        self.label_pythonpath["text"] = "Chemin vers l'interpreteur python: " + self.pythonpath
        showinfo("Modification de l'interpreteur python", "CNNTrainer va quitter, lorsque vous le rouvrirez il s'executera normalement sur le bon interpreteur")
        sys.exit()

    def choose_model(self):
        temp = askopenfilename(title="Choisir un modèle de réseau de neurones convolutionnel",filetypes=[('h5 files','.h5'),('all files','.*')])
        if(os.path.isfile(temp) and (os.path.splitext(temp)[1] == ".h5" or os.path.splitext(temp)[1] == ".tflite")):
            self.modelpath = temp
            self.label_modele_repris["text"] = ntpath.basename(temp)
        else:
            showerror("Fichier invalide", "Le fichier que vous indiquez n'est pas valide !")

    def choose_test_model(self):
        temp = askopenfilename(title="Choisir un modèle de réseau de neurones convolutionnel",filetypes=[('h5 files','.h5'),('all files','.*')])
        if(os.path.isfile(temp) and os.path.splitext(temp)[1] == ".h5"):
            self.trainedmodelpath = temp
            self.label_test_model["text"] = ntpath.basename(temp)
            if self.cnn_a_tester is None:
                #showerror("Probleme", "Le CNN n'a pas été chargé jusque là")
                self.cnn_a_tester = CNN_a_tester(temp)
            else:
                self.cnn_a_tester.actualiser(temp)
            #self.cnn_a_tester.path_to_model_weights = temp
        else:
            showerror("Fichier invalide", "Le fichier que vous indiquez n'est pas valide !")

    def actualiser_classes_test(self):
        self.liste_classes = []
        self.liste_classes = self.texte_classes.get("1.0",END).split("\n")
        self.liste_classes.remove('')
        self.nb_classes = len(self.liste_classes)
        self.label_nb_classes_test["text"] = str(self.nb_classes) + " classes trouvées"
        if self.cnn_a_tester is None:
            self.cnn_a_tester = CNN_a_tester(self.trainedmodelpath)
        #self.cnn_a_tester.nb_classes = self.nb_classes
        self.cnn_a_tester.categories = self.liste_classes
        #self.cnn_a_tester.actualiser()
        #print(self.liste_classes)

    def choose_test_image(self):
        temp = askopenfilename(title="Choisir un modèle de réseau de neurones convolutionnel",filetypes=[('all files','.*'),('péaingé','.png'),('jipégé','.jpg'),('jife','.gif')])
        if(os.path.isfile(temp)):
            if self.cnn_a_tester is None:
                showerror("Probleme", "Le CNN n'a pas été chargé jusque là")
                self.cnn_a_tester = CNN_a_tester(self.trainedmodelpath)
            reponse, suretee, img = self.cnn_a_tester.predict(temp)
            self.label_test_image["text"] = "Image : " + temp
            self.label_test_reponse["text"] = reponse
            self.label_test_suretee["text"] = str(suretee) + " %"
            """
            photo = PhotoImage(file=temp)
            self.test_image_cadre.create_image(200, 200, image=photo)
            """
            #self.logocadre.place(x=200, y=200, anchor=NW)
        else:
            showerror("Image invalide", "L'image que vous indiquez n'est pas valide !")


class CNN_a_tester():

    def __init__(self, path_to_model):
        import tensorflow as tf
        import glob
        from keras import __version__
        from keras.applications.inception_v3 import InceptionV3
        from keras.applications.inception_resnet_v2 import InceptionResNetV2
        from keras.applications.resnet50 import ResNet50
        from keras.preprocessing.image import ImageDataGenerator
        from keras.optimizers import SGD
        from keras.models import load_model

        self.config = tf.ConfigProto()
        self.config.gpu_options.allow_growth = True
        self.session = tf.Session(config=self.config)

        self.target_size = (299, 299)
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
        #print(preds[0])
        print(preds[0])

        for categorie, proba in zip(self.categories, preds[0]):
            #print(categorie + " : " + str(proba))
            preds_et_probas[categorie] = proba
        liste_triee = sorted(preds_et_probas, key=preds_et_probas.__getitem__, reverse=True)
        #print(liste_triee)
        suretee = max(preds[0]) * 100
        #print("suretee" + str(suretee))

        return liste_triee[0], suretee, img
        """
        except:
            print("Erreur !")
            return "", 0
        """


fenetre = Tk()
fenetre.title("CNNTrainer")
interface = Interface(fenetre)

interface.mainloop()
interface.destroy()