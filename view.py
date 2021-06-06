import traceback
from tkinter.filedialog import *
from tkinter import ttk as tkk
from tkinter.messagebox import showerror, showinfo
import ntpath

from logger import Logger
from thread_with_trace import Thread_with_trace

import sys
from training import *
from cnn import CNN

class Interface(Frame):
    """Notre fenêtre principale.
    Tous les widgets sont stockés comme attributs de cette fenêtre."""

    def __init__(self, fenetre, **kwargs):
        VIEW_DISABLED = DISABLED
        VIEW_NORMAL = NORMAL

        fenetre.config(height=WINDOW_HEIGHT, width=WINDOW_WIDTH)
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

        temp = "asupp"
        if (os.path.isfile(temp)):
            self.pythonpath = temp

        # Création de nos widgets
        self.logo = PhotoImage(file=LOGO_PATH)
        self.logocadre = Canvas(self, width=867, height=75)
        self.logocadre.create_image(0, 0, anchor=NW, image=self.logo)
        self.logocadre.place(x=200, y=200, anchor=NW)
        self.logocadre.pack()

        self.n = tkk.Notebook(self)
        self.f1 = Frame(self.n)  # first page, which would get widgets gridded into it
        self.f2 = Frame(self.n)  # second page
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

        # self.filepath = askopenfilename(title="Ouvrir une image",filetypes=[('png files','.png'),('all files','.*')])
        # self.photo = PhotoImage(file=self.filepath)
        self.bouton_choose_model = Button(self.continuer_entraienment, text="Choisir un modèle",
                                          command=self.choose_model, state=DISABLED)
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

        self.liste_modeles = CONFIG_KEYS(["models"])
        self.menu_deroulant_valeur_modeles = StringVar(value=self.liste_modeles[0])
        self.menu_deroulant_modeles = tkk.Combobox(self.base, textvariable=self.menu_deroulant_valeur_modeles)
        self.menu_deroulant_modeles.grid(row=0, column=1)
        self.menu_deroulant_modeles.bind('>', self.on_value_change)

        self.menu_deroulant_modeles['values'] = self.liste_modeles

        self.label_batch_size = Label(self.base, text="Taille batch: ")
        self.label_batch_size.grid(row=1)

        self.menu_deroulant_valeur_batch_size = StringVar(value='32')
        self.menu_deroulant_batch_size = tkk.Combobox(self.base, textvariable=self.menu_deroulant_valeur_batch_size)
        self.menu_deroulant_batch_size.grid(row=1, column=1)
        self.menu_deroulant_batch_size.bind('>', self.on_value_change)
        self.liste_batch_size = CONFIG["batch_sizes"]
        self.menu_deroulant_batch_size['values'] = self.liste_batch_size

        """
        self.liste_types_modeles = ('Inception V3', 'ResNet')
        self.texte_liste_types_modeles = StringVar()
        self.texte_liste_types_modeles.set(self.liste_types_modeles[0])
        self.menu_deroulant_modeles = OptionMenu(self.types_modele, self.texte_liste_types_modeles, *self.liste_types_modeles)
        self.menu_deroulant_modeles.pack(side="right")
        """

        self.valeur_changer_nb_l_to_f = IntVar()
        self.case_changer_nb_l_to_f = Checkbutton(self.base,
                                                  text="Changer nb layers to freeze",
                                                  variable=self.valeur_changer_nb_l_to_f,
                                                  command=self.faire_changer_nb_l_to_f)
        self.case_changer_nb_l_to_f.grid(row=2, columnspan=2)
        self.case_changer_nb_l_to_f.select()

        self.label_layers_to_freeze = Label(self.base, text="Nb layers to freeze: ")
        self.label_layers_to_freeze.grid(row=3)

        self.menu_deroulant_valeur_layers_to_freeze = StringVar(value='200')
        self.menu_deroulant_layers_to_freeze = tkk.Combobox(self.base,
                                                            textvariable=self.menu_deroulant_valeur_layers_to_freeze)
        self.menu_deroulant_layers_to_freeze.grid(row=3, column=1)
        self.menu_deroulant_layers_to_freeze.bind('>', self.on_value_change)
        self.liste_layers_to_freeze = CONFIG["nb_layers_to_freeze"]
        self.menu_deroulant_layers_to_freeze['values'] = self.liste_layers_to_freeze

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
        self.menu_deroulant_pas_preentrainement = tkk.Combobox(self.pre_entrainement,
                                                               textvariable=self.menu_deroulant_valeur_pas_preentrainement)
        self.menu_deroulant_pas_preentrainement.grid(row=1, column=1)
        self.menu_deroulant_pas_preentrainement.bind('>', self.on_value_change)
        self.liste_pas_preentrainement = CONFIG["pas_propositions"]
        self.menu_deroulant_pas_preentrainement['values'] = self.liste_pas_preentrainement
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
        self.menu_deroulant_epoch_preentrainement = tkk.Combobox(self.pre_entrainement,
                                                                 textvariable=self.menu_deroulant_valeur_epoch_preentrainement)
        self.menu_deroulant_epoch_preentrainement.grid(row=2, column=1)
        self.menu_deroulant_epoch_preentrainement.bind('>', self.on_value_change)
        self.liste_epoch_preentrainement = CONFIG["epoch_propositions"]
        self.menu_deroulant_epoch_preentrainement['values'] = self.liste_epoch_preentrainement

        self.label_optimiseur_preentrainement = Label(self.pre_entrainement, text="Optimiseur: ")
        self.label_optimiseur_preentrainement.grid(row=3)

        self.menu_deroulant_valeur_optimiseur_preentrainement = StringVar(value='RMSprop')
        self.menu_deroulant_optimiseur_preentrainement = tkk.Combobox(self.pre_entrainement,
                                                                      textvariable=self.menu_deroulant_valeur_optimiseur_preentrainement)
        self.menu_deroulant_optimiseur_preentrainement.grid(row=3, column=1)
        self.menu_deroulant_optimiseur_preentrainement.bind('>', self.on_value_change)
        self.liste_optimiseur_preentrainement = CONFIG["optimizers"]
        self.menu_deroulant_optimiseur_preentrainement['values'] = self.liste_optimiseur_preentrainement

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
        self.menu_deroulant_pas_entrainement = tkk.Combobox(self.entrainement,
                                                            textvariable=self.menu_deroulant_valeur_pas_entrainement)
        self.menu_deroulant_pas_entrainement.grid(row=1, column=1)
        self.menu_deroulant_pas_entrainement.bind('>', self.on_value_change)
        self.liste_pas_entrainement = CONFIG["pas_propositions"]
        self.menu_deroulant_pas_entrainement['values'] = self.liste_pas_entrainement

        self.label_epoch_entrainement = Label(self.entrainement, text="Nb epoch: ")
        self.label_epoch_entrainement.grid(row=2)

        self.menu_deroulant_valeur_epoch_entrainement = StringVar(value='50')
        self.menu_deroulant_epoch_entrainement = tkk.Combobox(self.entrainement,
                                                              textvariable=self.menu_deroulant_valeur_epoch_entrainement)
        self.menu_deroulant_epoch_entrainement.grid(row=2, column=1)
        self.menu_deroulant_epoch_entrainement.bind('>', self.on_value_change)
        self.liste_epoch_entrainement = CONFIG["epoch_propositions"]
        self.menu_deroulant_epoch_entrainement['values'] = self.liste_epoch_entrainement

        self.label_optimiseur_entrainement = Label(self.entrainement, text="Optimiseur: ")
        self.label_optimiseur_entrainement.grid(row=3)

        self.menu_deroulant_valeur_optimiseur_entrainement = StringVar(value='Adam')
        self.menu_deroulant_optimiseur_entrainement = tkk.Combobox(self.entrainement,
                                                                   textvariable=self.menu_deroulant_valeur_optimiseur_entrainement)
        self.menu_deroulant_optimiseur_entrainement.grid(row=3, column=1)
        self.menu_deroulant_optimiseur_entrainement.bind('>', self.on_value_change)
        self.liste_optimiseur_entrainement = CONFIG["optimizers"]
        self.menu_deroulant_optimiseur_entrainement['values'] = self.liste_optimiseur_entrainement

        ### --- Dataset --- ###

        self.dataset = LabelFrame(self.f1, text="Dataset")
        self.dataset.pack(fill=X, side=TOP, expand="yes", padx=self.padding, pady=self.padding)

        self.label_dataset_train = Label(self.dataset, text="0 photo, 0 classe")
        self.label_dataset_train.grid(row=1)

        self.bouton_choose_dataset_train = Button(self.dataset, text="Dossier d'entrainement",
                                                  command=self.choose_dataset_train)
        self.bouton_choose_dataset_train.grid(row=1, column=1)

        self.label_dataset_validation = Label(self.dataset, text="0 photo, 0 classe")
        self.label_dataset_validation.grid(row=2)

        self.bouton_choose_dataset_validation = Button(self.dataset, text="Dossier de validation",
                                                       command=self.choose_dataset_validation)
        self.bouton_choose_dataset_validation.grid(row=2, column=1)

        ### --- Lancer l'entrainement --- ###

        self.cadre_lancer_entrainement = Frame(self.f1, borderwidth=20, relief=FLAT)
        self.cadre_lancer_entrainement.pack(fill=X, side=TOP, expand="yes", padx=self.padding, pady=self.padding)

        self.bouton_lancer_entrainement = Button(self.cadre_lancer_entrainement, text="Lancer l'entrainement !",
                                                 fg="green", command=self.verification_et_lancement)
        self.bouton_lancer_entrainement.pack(side="left")

        self.bouton_stopper_entrainement = Button(self.cadre_lancer_entrainement, text="Stopper l'entrainement",
                                                  fg="red", command=self.stopper_entrainement)
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

        # command = "C:\\Users\\EnzoGamer\\AppData\\Local\\conda\\conda\\envs\\tf_gpu\\python.exe \"C:\\Users\\EnzoGamer\\Desktop\\PROJET IA\\cnn trainer\\CNNTrainer\\script.py\""

        # t = Thread(target = lambda: leterminal(command, self.text))
        # t.start()

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

        # self.filepath = askopenfilename(title="Ouvrir une image",filetypes=[('png files','.png'),('all files','.*')])
        # self.photo = PhotoImage(file=self.filepath)
        self.bouton_choose_test_model = Button(self.test_model, text="Choisir un modèle",
                                               command=self.choose_test_model)
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

        self.bouton_lancer_entrainement = Button(self.cadre_actualiser_classes, text="Actualiser",
                                                 command=self.actualiser_classes_test)
        self.bouton_lancer_entrainement.pack(side="bottom")

        ### --- Test de l'image --- ###

        self.test_image = LabelFrame(self.f2, text="Test")
        self.test_image.grid(row=0, column=1, rowspan=20, padx=self.padding, pady=self.padding)

        self.test_image_cadre = Canvas(self.test_image, width=500, height=500)
        # self.test_image_cadre.create_image(0, 0, anchor=NW)
        self.test_image_cadre.grid(row=0, columnspan=2)

        self.label_test_image = Label(self.test_image, text="Image : Aucun fichier ")
        self.label_test_image.grid(row=1)

        # self.filepath = askopenfilename(title="Ouvrir une image",filetypes=[('png files','.png'),('all files','.*')])
        # self.photo = PhotoImage(file=self.filepath)
        self.bouton_choose_test_image = Button(self.test_image, text="Choisir une image",
                                               command=self.choose_test_image)
        self.bouton_choose_test_image.grid(row=1, column=1)

        self.label_test_reponse = Label(self.test_image, text="...")
        self.label_test_reponse.grid(row=2)

        self.label_test_suretee = Label(self.test_image, text="0 %")
        self.label_test_suretee.grid(row=3)

        self.label_pythonpath = Label(self, text="Chemin vers l'interpreteur python: " + self.pythonpath)
        self.label_pythonpath.pack(side="left")

        self.bouton_cliquer = Button(self, text="Changer d'interpreteur python", fg="red",
                                     command=self.change_pyhton_path)
        self.bouton_cliquer.pack(side="right")

    def verification_et_lancement(
            self):  ### --- --- On ne peut pas juste faire un include sinon on a pas la sortie du terminal --- --- ###
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
        change_nb_l_to_f = True

        erronne = False
        erreurs = []

        try:

            if (self.nb_classes_train == 0 and self.nb_classes_validation == 0):
                erronne = True
                erreurs.append("Aucune classe n'est fournie")
            elif (self.nb_classes_train != self.nb_classes_validation):
                erronne = True
                erreurs.append("Le nombre de classes des dossiers de validation et d'entrainement sont différents")
            else:
                nb_classes = self.nb_classes_train
                dir_train = self.trainpath
                for root, dirs, files in os.walk(self.trainpath):
                    dir_train_nb_fic += len(files)
                # dir_train_nb_fic = int(len(next(os.walk(self.trainpath))[1]))
                dir_validation = self.validationpath
                for root, dirs, files in os.walk(self.validationpath):
                    dir_validation_nb_fic += len(files)
                # dir_validation_nb_fic = int(len(next(os.walk(self.validationpath))[1]))

            try:
                batch_size = int(self.menu_deroulant_valeur_batch_size.get())
            except:
                erronne = True
                erreurs.append("La valeur donnée pour la taille du batch (tampon) n'est pas un entier valide")

            if (self.continue_train.get() == 1):
                reprise = True
                path_modele = self.modelpath

            if self.menu_deroulant_valeur_modeles.get() in CONFIG["models"].keys():
                type_modele = self.menu_deroulant_valeur_modeles.get()
            else:
                erronne = True
                erreurs.append(
                    "Le modele demandé n'est pas supporté, veuillez entrer un modele parmis ceux proposés (il est possible que vous ayez fait une erreur de frappe)")

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
                    erreurs.append(
                        "L'optimiseur demandé n'est pas supporté, veuillez entrer un optimiseur parmis ceux proposés (il est possible que vous ayez fait une erreur de frappe)")

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
                erreurs.append(
                    "L'optimiseur demandé n'est pas supporté, veuillez entrer un optimiseur parmis ceux proposés (il est possible que vous ayez fait une erreur de frappe)")

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

            if self.valeur_changer_nb_l_to_f.get() == 1:
                change_nb_l_to_f = True
            else:
                change_nb_l_to_f = False

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

            sys.stdout = Logger(self.text)
            sys.stderr = Logger(self.text)

            # On va se baser sur les différentes sorties pour calculer l'avancement de la barre de progression
            sys.stdout.set_training_bar(self.processing_bar)
            sys.stderr.set_training_bar(self.processing_bar)

            self.thread_entrainement = Thread_with_trace(
                target=lambda: run_training(self.bouton_lancer_entrainement, nb_layers_to_freeze, change_nb_l_to_f,
                                            reprise, reprise_poids, finetuning_partiel, path_modele, nb_classes,
                                            dir_train, dir_train_nb_fic, dir_validation, dir_validation_nb_fic,
                                            preentrainement, nb_epoch_preentrainement, pas_preentrainement,
                                            nb_epoch_entrainement, pas_entrainement, batch_size, type_modele,
                                            optimiseur_preentrainement, optimiseur_entrainement))
            self.thread_entrainement.start()

    def stopper_entrainement(self):
        self.thread_entrainement.kill()
        self.thread_entrainement.join()

        if not self.thread_entrainement.isAlive():
            print("=== === Entrainement stoppé === ===")

    def choose_dataset_train(self):
        temp = askdirectory(title="Dossier contenant les images à utiliser lors de l'entrainement")
        if (os.path.exists(temp) and int(len(next(os.walk(temp))[1])) > 0):
            dir_train_nb_fic = 0
            for root, dirs, files in os.walk(temp):
                dir_train_nb_fic += len(files)
            # print(dir_train_nb_fic)

            if (dir_train_nb_fic > 0):
                self.trainpath = temp
                self.nb_classes_train = int(len(next(os.walk(temp))[1]))
                self.label_dataset_train["text"] = str(dir_train_nb_fic) + " photos, " + str(
                    self.nb_classes_train) + " classes"
            else:
                showerror("Aucune photo trouvée",
                          "le dossier d'entrainement que vous avez donné semble contenir des classes mais ne contient aucune photo !")
        elif (int(len(next(os.walk(temp))[1])) == 0):
            showerror("Aucune classe trouvée",
                      "le dossier d'entrainement que vous avez donné ne contient aucune classe !")
            self.label_dataset_train["text"] = "0 photo"
        else:
            showerror("Chemin invalide", "le chemin que vous avez donné n'est pas valide !")

    def choose_dataset_validation(self):
        temp = askdirectory(
            title="Dossier contenant les images à utiliser pour vérifier la performance du CCN pendant l'entrainement")
        if (os.path.exists(temp) and int(len(next(os.walk(temp))[1])) > 0):
            dir_validation_nb_fic = 0
            for root, dirs, files in os.walk(temp):
                dir_validation_nb_fic += len(files)
            # print(dir_validation_nb_fic)

            if (dir_validation_nb_fic > 0):
                self.validationpath = temp
                self.nb_classes_validation = int(len(next(os.walk(temp))[1]))
                self.label_dataset_validation["text"] = str(dir_validation_nb_fic) + " photos, " + str(
                    self.nb_classes_validation) + " classes"
            else:
                showerror("Aucune photo trouvée",
                          "le dossier de validation que vous avez donné semble contenir des classes mais ne contient aucune photo !")
        elif (int(len(next(os.walk(temp))[1])) == 0):
            showerror("Aucune classe trouvée",
                      "le dossier de validation que vous avez donné ne contient aucune classe !")
            self.label_dataset_validation["text"] = "0 photo"
        else:
            showerror("Chemin invalide", "le chemin que vous avez donné n'est pas valide !")

    def on_value_change(self):
        print("méthode on_value_change() utilisée")

    def griser_interface_preentrainement(self):
        if (self.active_pre_entrainement.get() == 1):
            self.menu_deroulant_pas_preentrainement.config(state=NORMAL)
            self.menu_deroulant_epoch_preentrainement.config(state=NORMAL)
            self.menu_deroulant_optimiseur_preentrainement.config(state=NORMAL)
        else:
            self.menu_deroulant_pas_preentrainement.config(state=DISABLED)
            self.menu_deroulant_epoch_preentrainement.config(state=DISABLED)
            self.menu_deroulant_optimiseur_preentrainement.config(state=DISABLED)

    def griser_bouton_continuer(self):
        if (self.continue_train.get() == 1):
            self.bouton_choose_model.config(state=NORMAL)
            self.case_only_weights.config(state=NORMAL)
            self.case_finetuning_partiel.config(state=NORMAL)
            if self.only_weights.get() == 1:
                self.menu_deroulant_modeles.config(state=NORMAL)
            else:
                self.menu_deroulant_modeles.config(state=DISABLED)
        else:
            self.bouton_choose_model.config(state=DISABLED)
            self.case_only_weights.config(state=DISABLED)
            self.case_finetuning_partiel.config(state=DISABLED)
            self.menu_deroulant_modeles.config(state=NORMAL)

    def ne_contient_que_les_poids(self):
        if (self.only_weights.get() == 1):
            self.menu_deroulant_modeles.config(state=NORMAL)
        else:
            self.menu_deroulant_modeles.config(state=DISABLED)

    def faire_finetuning_partiel(self):
        pass

    def change_pyhton_path(self):
        temp = askopenfilename(title="Indiquer l'interpreteur python à utiliser",
                               filetypes=[('python', '.*'), ('all files', '.*')])
        if (os.path.isfile(temp)):
            self.pythonpath = temp
            change_path(temp)
            # changer_xml(balise="pyhton_path", attribut="path", valeur=temp)
        else:
            print("le chemin que vous venez de donner n'est pas valide")
        print("affichage de pythonpath :")
        print(self.pythonpath)
        print("modification du label de texte ..")
        self.label_pythonpath["text"] = "Chemin vers l'interpreteur python: " + self.pythonpath
        showinfo("Modification de l'interpreteur python",
                 "CNNTrainer va quitter, lorsque vous le rouvrirez il s'executera normalement sur le bon interpreteur")
        sys.exit()

    def choose_model(self):
        temp = askopenfilename(title="Choisir un modèle de réseau de neurones convolutionnel",
                               filetypes=[('h5 files', '.h5'), ('all files', '.*')])
        if (os.path.isfile(temp) and (os.path.splitext(temp)[1] == ".h5" or os.path.splitext(temp)[1] == ".tflite")):
            self.modelpath = temp
            self.label_modele_repris["text"] = ntpath.basename(temp)
        else:
            showerror("Fichier invalide", "Le fichier que vous indiquez n'est pas valide !")

    def choose_test_model(self):
        temp = askopenfilename(title="Choisir un modèle de réseau de neurones convolutionnel",
                               filetypes=[('h5 files', '.h5'), ('all files', '.*')])
        if (os.path.isfile(temp) and os.path.splitext(temp)[1] == ".h5"):
            self.trainedmodelpath = temp
            self.label_test_model["text"] = ntpath.basename(temp)
            if self.cnn_a_tester is None:
                # showerror("Probleme", "Le CNN n'a pas été chargé jusque là")
                self.cnn_a_tester = CNN(self.trainedmodelpath)
                if self.cnn_a_tester is None:
                    showerror("Probleme", "Le CNN n'a pas pu etre chargé")
            else:
                self.cnn_a_tester.actualiser(self.trainedmodelpath)
            # self.cnn_a_tester.path_to_model_weights = temp
        else:
            showerror("Fichier invalide", "Le fichier que vous indiquez n'est pas valide !")

    def actualiser_classes_test(self):
        self.liste_classes = []
        self.liste_classes = self.texte_classes.get("1.0", END).split("\n")
        self.liste_classes.remove('')
        self.nb_classes = len(self.liste_classes)
        self.label_nb_classes_test["text"] = str(self.nb_classes) + " classes trouvées"
        """
        if self.cnn_a_tester is None:
            self.cnn_a_tester = CNN_a_tester(self.trainedmodelpath)
        """
        # self.cnn_a_tester.nb_classes = self.nb_classes
        self.cnn_a_tester.categories = self.liste_classes
        # self.cnn_a_tester.actualiser()
        # print(self.liste_classes)

    def choose_test_image(self):
        temp = askopenfilename(title="Choisir un modèle de réseau de neurones convolutionnel",
                               filetypes=[('all files', '.*'), ('péaingé', '.png'), ('jipégé', '.jpg'),
                                          ('jife', '.gif')])
        if (os.path.isfile(temp)):
            if self.cnn_a_tester is None:
                showerror("Probleme", "Le CNN n'a pas été chargé jusque là")
                self.cnn_a_tester = CNN(self.trainedmodelpath)
            reponse, suretee, img = self.cnn_a_tester.predict(temp)
            self.label_test_image["text"] = "Image : " + temp
            self.label_test_reponse["text"] = reponse
            self.label_test_suretee["text"] = str(suretee) + " %"
            """
            photo = PhotoImage(file=temp)
            self.test_image_cadre.create_image(200, 200, image=photo)
            """
            # self.logocadre.place(x=200, y=200, anchor=NW)
        else:
            showerror("Image invalide", "L'image que vous indiquez n'est pas valide !")

    def faire_changer_nb_l_to_f(self):
        if (self.valeur_changer_nb_l_to_f.get() == 1):
            self.menu_deroulant_layers_to_freeze.config(state=NORMAL)
        else:
            self.menu_deroulant_layers_to_freeze.config(state=DISABLED)

