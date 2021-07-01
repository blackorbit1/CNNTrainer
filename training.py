import time

from global_variables import *
from ml_utils import *
from utils import *

import os

def run_training(bouton_lancer_entrainement, nb_layers_to_freeze, change_nb_l_to_f, reprise, reprise_poids,
                 finetuning_partiel, dir_modele, nb_classes, dir_train, dir_train_nb_fic, dir_validation,
                 dir_validation_nb_fic, preentrainement, nb_epoch_preentrainement, pas_preentrainement,
                 nb_epoch_entrainement, pas_entrainement, batch_size, type_modele, optimiseur_preentrainement,
                 optimiseur_entrainement):
    bouton_lancer_entrainement.config(state=VIEW_DISABLED)
    print("\n")
    print(os.getcwd())

    from tensorflow.keras.callbacks import TensorBoard
    from tensorflow.keras.applications.inception_v3 import preprocess_input
    from tensorflow.keras.models import Model
    from tensorflow.keras.layers import Dense
    from tensorflow.keras.preprocessing.image import ImageDataGenerator



    from tensorflow.keras.callbacks import ModelCheckpoint



    print("\nChargement des packages terminés\n")

    if not reprise: dir_modele = generate_model_filename()

    print("\nLe directory du fichier contenant le CNN qui sera enregistré : " + dir_modele + "\n")

    print("\nLancement de l'entrainement ...\n")

    IM_WIDTH = CONFIG["models"][type_modele]["max_width"]
    IM_HEIGHT = CONFIG["models"][type_modele]["max_height"]

    # data prep
    train_datagen = ImageDataGenerator(
        preprocessing_function=preprocess_input,
        rotation_range=180,
        width_shift_range=0.4,
        height_shift_range=0.4,
        shear_range=0.3,
        zoom_range=0.5,
        horizontal_flip=True,
        vertical_flip=True,
        channel_shift_range=30,
        brightness_range=[0.2, 1.0]
    )

    train_generator = train_datagen.flow_from_directory(
        dir_train,
        target_size=(IM_WIDTH, IM_HEIGHT),
        batch_size=batch_size,
    )

    test_datagen = ImageDataGenerator(
        preprocessing_function=preprocess_input,
        rotation_range=30,
        width_shift_range=0.1,
        height_shift_range=0.1,
        shear_range=0.1,
        zoom_range=0.1,
        horizontal_flip=True
    )

    validation_generator = test_datagen.flow_from_directory(
        dir_validation,
        target_size=(IM_WIDTH, IM_HEIGHT),
        batch_size=batch_size,
    )

    if reprise:
        if reprise_poids:
            base_model = get_model(model_name=type_modele, width=IM_WIDTH, height=IM_HEIGHT, nb_layers=3)
            model = add_new_last_layer(base_model, nb_classes)
            model.load_weights(dir_modele)
        else:
            from tensorflow.keras.models import load_model
            print("\nchargement du modele ...")
            model = load_model(dir_modele)
            print("chargement du modele terminé\n")
            if finetuning_partiel:
                model.layers.pop()

                # add a global spatial average pooling layer
                x = model.layers[-1].output

                predictions = Dense(nb_classes, activation='softmax', name="dense_final_1")(x)
                model = Model(inputs=model.input, outputs=predictions)
    else:
        base_model = get_model(model_name=type_modele, width=IM_WIDTH, height=IM_HEIGHT, nb_layers=3)
        model = add_new_last_layer(base_model, nb_classes)

    tensorboard = TensorBoard(log_dir=TENSORBOARD_LOG_PATH, histogram_freq=0, write_graph=True, write_images=False)


    if preentrainement:
        # transfer learning
        for layer in base_model.layers:
            layer.trainable = False

        optimizer = get_optimiser(pas_preentrainement, optimiseur_preentrainement)
        model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])

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
            # nb_val_samples=dir_validation_nb_fic,
            # class_weight='auto'
        )

    # fine-tuning
    if change_nb_l_to_f:
        for layer in model.layers[:nb_layers_to_freeze]:
            layer.trainable = False
        for layer in model.layers[nb_layers_to_freeze:]:
            layer.trainable = True

    optimizer = get_optimiser(pas_preentrainement, optimiseur_preentrainement)
    model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])



    filepath = dir_modele + "-wi-{epoch:02d}-{val_acc:.2f}.hdf5"
    checkpoint = ModelCheckpoint(filepath, monitor='val_acc', verbose=1, save_best_only=True, mode='max')
    callbacks_list = [checkpoint, checkpoint, tensorboard]

    print("\n\n--- --- Lancement de l'entrainement --- ---\n")

    history_ft = model.fit_generator(
        train_generator,  # generateur de nouvelles image d'entrainement
        #samples_per_epoch=dir_train_nb_fic,  # nb de fichiers d'entrainement
        epochs=nb_epoch_entrainement,  # nb de cycles d'entrainement
        workers=1,  # nb d'user travaillant dessus (laisser 1 si GPU)
        use_multiprocessing=False,  # laisser False si GPU TODO : à retenir si mise en place d'une option CPU / GPU
        # steps_per_epoch=dir_train_nb_fic // batch_size,         # nb fic entrainement / taille tampon
        validation_steps=dir_validation_nb_fic // batch_size,  # nb fic validation / taille tampon
        validation_data=validation_generator,  # generateur de nouvelles image de validation
        # nb_val_samples=dir_validation_nb_fic,                  # nb fichiers de validation (laisser en com)
        #class_weight="auto",
        callbacks=callbacks_list,
        shuffle=True
        # verbose=2
    )

    time.sleep(10000)


    print("""
        ╔══════════════════════════════════════════════════════╗
        ║          Entrainement terminé, à bientot !           ║
        ╚══════════════════════════════════════════════════════╝
    """)

    print("\n\nEnregistrement du fichier " + dir_modele + " ....")
    model.save(dir_modele)  # always save your weights after training or during training
    print("fichier enregistré !")
    bouton_lancer_entrainement.config(state=VIEW_NORMAL)
