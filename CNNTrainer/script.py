"""
Auteur : DUTRA Enzo (10/2/2019)
"""
import sys
import os

RMSPROP_LR_DEFAULT = 0.001
ADAM_LR_DEFAULT = 0.001
BATCH_SIZE_DEFAULT = 32
IM_WIDTH, IM_HEIGHT = 150, 150 # fixed size for InceptionV3
NB_IV3_LAYERS_TO_FREEZE = 172 # nb de couches d'inception V3 à laisser statique

user_input = ''

def isfloat(value):
  try:
    float(value)
    return True
  except ValueError:
    return False

def recevoir_parametres_CNN():
    nb_classes = 0
    dir_train = ""
    dir_train_nb_fic = 0
    dir_validation = ""
    dir_validation_nb_fic = 0
    preentrainement = False
    nb_epoch_preentrainement = 0
    pas_preentrainement = 0
    nb_epoch_entrainement = 0
    pas_entrainement = 0
    batch_size = 0

    entree_user = ""

    while not entree_user.isnumeric():
        entree_user = input("Combiens de classe doit contenir votre CNN ?\n> ")
    nb_classes = entree_user

    while True:
        while not os.path.exists(dir_train):
            entree_user = input("Donnez le chemin vers le dossier d'entrainement :\n> ")
            dir_train = entree_user
        if(int(nb_classes) != int(len(next(os.walk(dir_train))[1]))):
            print("Le dossier que vous avez donné contient " + str(len(next(os.walk(dir_train))[1])) + " dossiers alors que vous avez demandé " + str(nb_classes) + " classes")
        else :
            for root, dirs, files in os.walk(dir_train):
                dir_train_nb_fic += len(files)
            print("Nombre de fichiers trouvés : " + str(dir_train_nb_fic))
            if input("valider ? ('o' = oui, 'n' = non)\n> ") == 'o' : break
        dir_train = ""
        dir_train_nb_fic = 0

    while True:
        while not os.path.exists(dir_validation):
            entree_user = input("Donnez le chemin vers le dossier de validation :\n> ")
            dir_validation = entree_user
        if(int(nb_classes) != int(len(next(os.walk(dir_validation))[1]))):
            print("Le dossier que vous avez donné contient " + str(len(next(os.walk(dir_validation))[1])) + " dossiers alors que vous avez demandé " + str(nb_classes) + " classes")
        else :
            for root, dirs, files in os.walk(dir_validation):
                dir_validation_nb_fic += len(files)
            print("Nombre de fichiers trouvés : " + str(dir_validation_nb_fic))
            if input("valider ? ('o' = oui, 'n' = non)\n> ") == 'o' : break
        dir_validation = ""
        dir_validation_nb_fic = 0

    while entree_user != 'o' and entree_user != 'n':
        entree_user = input("Voulez vous faire un préentrainement ? ('o' = oui, 'n' = non)\n> ")
    if entree_user != 'n' :
        preentrainement = True
        print("--- --- Paramétrage du préentrainement --- ---")

        while not entree_user.isnumeric():
            entree_user = input("Combiens de fois voulez vous repasser le dataset lors de l'entrainement de la derniere couche ?\n> ")
        nb_epoch_preentrainement = int(entree_user)
        entree_user = ""

        while not isfloat(entree_user):
            entree_user = input("Quel pas voulez vous donner lors de l'entrainement de la derniere couche ? (default : " + str(RMSPROP_LR_DEFAULT) + ")\n> ")
            if entree_user == "" : entree_user = RMSPROP_LR_DEFAULT
        pas_preentrainement = float(entree_user)
        entree_user = ""

    print("--- --- Paramétrage de l'entrainement --- ---")

    while not entree_user.isnumeric():
        entree_user = input("Combiens de fois voulez vous repasser le dataset lors du réentrainement d'inception V3 ?\n> ")
    nb_epoch_entrainement = int(entree_user)
    entree_user = ""

    while not isfloat(entree_user):
        entree_user = input("Quel pas voulez vous donner lors du réentrainement d'inception V3 ? (default : " + str(ADAM_LR_DEFAULT) + ")\n> ")
        if entree_user == "" : entree_user = ADAM_LR_DEFAULT
    pas_entrainement = float(entree_user)
    entree_user = ""

    while not isfloat(entree_user):
        entree_user = input("Quelle taille de batch / tampon voulez vous donner ? (default : " + str(BATCH_SIZE_DEFAULT) + ")\n> ")
        if entree_user == "" : entree_user = BATCH_SIZE_DEFAULT
    batch_size = int(entree_user)
    entree_user = ""

    return nb_classes, dir_train, dir_train_nb_fic, dir_validation, dir_validation_nb_fic, preentrainement, nb_epoch_preentrainement, pas_preentrainement, nb_epoch_entrainement, pas_entrainement, batch_size


def setup_to_transfer_learn(model, base_model, pas_preentrainement, optimiseur_preentrainement):
    """Freeze all layers and compile the model"""
    for layer in base_model.layers:
        layer.trainable = False
    if(optimiseur_preentrainement == "RMSprop"):
        optimiseur = RMSprop(lr=pas_preentrainement, rho=0.9, epsilon=None, decay=0.0)
        model.compile(optimizer=optimiseur, loss='categorical_crossentropy', metrics=['accuracy'])
    elif optimiseur_preentrainement == "Adam":
        optimiseur = Adam(lr=pas_preentrainement, beta_1=0.9, beta_2=0.999, epsilon=1e-8)
        model.compile(optimizer=optimiseur, loss='categorical_crossentropy', metrics=['accuracy'])
    elif optimiseur_preentrainement == "SGD":
        optimiseur = SGD(lr=pas_preentrainement, momentum=0.0, decay=0.0, nesterov=False)
        model.compile(optimizer=optimiseur, loss='categorical_crossentropy', metrics=['accuracy'])
    else:
        optimiseur = RMSprop(lr=pas_preentrainement, rho=0.9, epsilon=None, decay=0.0)
        model.compile(optimizer=optimiseur, loss='categorical_crossentropy', metrics=['accuracy'])



def add_new_last_layer(base_model, nb_classes):
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
    model = Model(inputs=base_model.input, outputs=predictions)
    return model


def setup_to_finetune(model, pas_entrainement, optimiseur_entrainement):
    """Freeze the bottom NB_IV3_LAYERS and retrain the remaining top layers.
    note: NB_IV3_LAYERS corresponds to the top 2 inception blocks in the inceptionv3 arch
    Args:
    model: keras model
    """
    for layer in model.layers[:NB_IV3_LAYERS_TO_FREEZE]:
        layer.trainable = False
    for layer in model.layers[NB_IV3_LAYERS_TO_FREEZE:]:
        layer.trainable = True
    if(optimiseur_entrainement == "RMSprop"):
        optimiseur = RMSprop(lr=pas_entrainement, rho=0.9, epsilon=None, decay=0.0)
        model.compile(optimizer=optimiseur, loss='categorical_crossentropy', metrics=['accuracy'])
    elif optimiseur_entrainement == "Adam":
        optimiseur = Adam(lr=pas_entrainement, beta_1=0.9, beta_2=0.999, epsilon=1e-8)
        model.compile(optimizer=optimiseur, loss='categorical_crossentropy', metrics=['accuracy'])
    elif optimiseur_entrainement == "SGD":
        optimiseur = SGD(lr=pas_entrainement, momentum=0.0, decay=0.0, nesterov=False)
        model.compile(optimizer=optimiseur, loss='categorical_crossentropy', metrics=['accuracy'])
    else:
        optimiseur = Adam(lr=pas_entrainement, beta_1=0.9, beta_2=0.999, epsilon=1e-8)
        model.compile(optimizer=optimiseur, loss='categorical_crossentropy', metrics=['accuracy'])
    #model.compile(optimizer=Adam(lr=pas_entrainement, beta_1=0.9, beta_2=0.999, epsilon=1e-8), loss='categorical_crossentropy', metrics=['accuracy'])


print("\n")
print(os.getcwd())

import glob
import argparse
#import matplotlib.pyplot as plt
from keras.applications.inception_v3 import InceptionV3, preprocess_input
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

reprise = True if args.reprise == "true" else False
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






# nom automatique du modele généré
dir_modele = "model_"
nb_modele_dir = 1
while os.path.isfile(dir_modele + str(nb_modele_dir) + ".h5"):
    nb_modele_dir += 1
dir_modele = dir_modele + str(nb_modele_dir) + ".h5"

# Si l'user avait demandé de continuer l'entrainement d'un CNN déjà existant
if user_input == 'c':
    while not os.path.isfile(dir_modele):
        entree_user = input("Donnez le chemin vers le modele que vous voulez continuer à entrainer :\n> ")
        dir_modele = entree_user

print("\nLe directory du fichier contenant le CNN qui sera enregistré : " + dir_modele + "\n")


print("\nLancement de l'entrainement ...\n")

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
test_datagen = ImageDataGenerator(
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

validation_generator = test_datagen.flow_from_directory(
    dir_validation,
    target_size=(IM_WIDTH, IM_HEIGHT),
    batch_size=batch_size,
)

# setup model
if(type_modele == "resnet"):
    base_model = ResNet50(weights='imagenet', include_top=False)
else:
    base_model = InceptionV3(weights='imagenet', include_top=False) #include_top=False excludes final FC layer
model = add_new_last_layer(base_model, nb_classes)

if reprise:
    model.load_weights(dir_modele)

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
setup_to_finetune(model, pas_entrainement, optimiseur_entrainement)

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
    steps_per_epoch=dir_train_nb_fic // batch_size,         # nb fic entrainement / taille tampon
    validation_steps=dir_validation_nb_fic // batch_size,   # nb fic validation / taille tampon
    validation_data=validation_generator,                   # generateur de nouvelles image de validation
    #nb_val_samples=dir_validation_nb_fic,                  # nb fichiers de validation (laisser en com)
    class_weight="auto",
    #verbose=2
)

print("""
    ╔══════════════════════════════════════════════════════╗
    ║          Entrainement terminé, à bientot !           ║
    ╚══════════════════════════════════════════════════════╝
""")

print("\n\nEnregistrement du fichier " + dir_modele + " ....")
model.save_weights(dir_modele)  # always save your weights after training or during training
