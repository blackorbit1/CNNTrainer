import subprocess


def import_or_install(package):
    try:
        __import__(package)
    except ImportError:
        subprocess.check_call(["python", '-m', 'pip', 'install', package]) # install pkg
        #pip.main(['install', package])
    except ModuleNotFoundError:
        print("le module n'a pas été trouvé, installation ...")
        subprocess.check_call(["python", '-m', 'pip', 'install', package]) # install pkg
        #pip.main(['install', package])

dependencies = ["xml", "lxml", "lxml.etree", "tkinter"]
for package in dependencies:
    import_or_install(package)

from tkinter import *
from tkinter.filedialog import *
from tkinter import ttk as tkk
from tkinter.messagebox import showerror
import tkinter as tk
from threading import Thread
import xml.etree.ElementTree as ET
from lxml import etree

def leterminal(command, terminal):
    p = subprocess.Popen(command, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE, bufsize=1, universal_newlines=True, shell = True, encoding="utf8")
    p.poll()

    while True:
        line = p.stdout.readline()
        terminal.insert(tk.END, line)
        terminal.see(tk.END)
        if not line and p.poll is not None: break

    while True:
        err = p.stderr.readline()
        terminal.insert(tk.END, err)
        terminal.see(tk.END)
        if not err and p.poll is not None: break

    terminal.insert(tk.END, "fin de l'execution du script python")

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

        self.trainpath = ""
        self.validationpath = ""
        self.pythonpath = ""
        self.nb_classes = 0
        self.nb_classes_train = 0
        self.nb_classes_validation = 0
        temp = voir_xml(balise="pyhton_path", attribut="path")
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
        self.text.config(font=("Source Code Pro", 10))
        self.text.pack(fill=Y, side=RIGHT)


        ### --- Reprise d'entrainement --- ###

        self.continuer_entraienment = LabelFrame(self.f1, text="Reprise d'entrainement")
        self.continuer_entraienment.pack(fill=X, side=TOP, expand="yes", padx=self.padding, pady=self.padding)

        self.continue_train = IntVar()
        self.case_continue_train = Checkbutton(self.continuer_entraienment,
                                               text="Continuer un entrainement",
                                               variable=self.continue_train,
                                               command=self.griser_bouton_continuer).pack()

        #self.filepath = askopenfilename(title="Ouvrir une image",filetypes=[('png files','.png'),('all files','.*')])
        #self.photo = PhotoImage(file=self.filepath)
        self.bouton_choose_model = Button(self.continuer_entraienment, text="Choisir un modèle", command=self.choose_model, state=DISABLED)
        self.bouton_choose_model.pack()


        ### --- Type de model --- ###

        self.types_modele = LabelFrame(self.f1, text="Type de modele")
        self.types_modele.pack(fill=X, side=TOP, expand="yes", padx=self.padding, pady=self.padding)

        self.label_nb_epoch = Label(self.types_modele, text="Type de modele: ")
        self.label_nb_epoch.pack(side="left")


        self.menu_deroulant_valeur_modeles = StringVar(value='Inception V3')
        self.menu_deroulant_modeles = tkk.Combobox(self.types_modele, textvariable=self.menu_deroulant_valeur_modeles)
        self.menu_deroulant_modeles.pack(side="right")
        self.menu_deroulant_modeles.bind('>', self.on_value_change)
        self.liste_modeles = ['Inception V3', 'ResNet']
        self.menu_deroulant_modeles['values'] =  self.liste_modeles

        """
        self.liste_types_modeles = ('Inception V3', 'ResNet')
        self.texte_liste_types_modeles = StringVar()
        self.texte_liste_types_modeles.set(self.liste_types_modeles[0])
        self.menu_deroulant_modeles = OptionMenu(self.types_modele, self.texte_liste_types_modeles, *self.liste_types_modeles)
        self.menu_deroulant_modeles.pack(side="right")
        """


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
        self.liste_optimiseur_preentrainement = ["RMSprop", "Adam", "SGD"]
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
        self.liste_optimiseur_entrainement = ["RMSprop", "Adam", "SGD"]
        self.menu_deroulant_optimiseur_entrainement['values'] =  self.liste_optimiseur_entrainement


        ### --- Dataset --- ###

        self.dataset = LabelFrame(self.f1, text="Dataset")
        self.dataset.pack(fill=X, side=TOP, expand="yes", padx=self.padding, pady=self.padding)

        self.label_dataset_train = Label(self.dataset, text="0 photo")
        self.label_dataset_train.grid(row=1)

        self.bouton_choose_dataset_train = Button(self.dataset, text="Dossier d'entrainement", command=self.choose_dataset_train)
        self.bouton_choose_dataset_train.grid(row=1, column=1)


        self.label_dataset_validation = Label(self.dataset, text="0 photo")
        self.label_dataset_validation.grid(row=2)

        self.bouton_choose_dataset_validation = Button(self.dataset, text="Dossier de validation", command=self.choose_dataset_validation)
        self.bouton_choose_dataset_validation.grid(row=2, column=1)


        ### --- Lancer l'entrainement --- ###

        self.cadre_lancer_entrainement = Frame(self.f1, borderwidth=20, relief=FLAT)
        self.cadre_lancer_entrainement.pack(fill=X, side=TOP, expand="yes", padx=self.padding, pady=self.padding)

        self.bouton_lancer_entrainement = Button(self.cadre_lancer_entrainement, text="Lancer l'entrainement !", command=self.choose_dataset_validation)
        self.bouton_lancer_entrainement.pack(side="bottom")

        """
        import tkinter.font
        for name in sorted(tkinter.font.families()):
            print(name)
        """


        command = "C:\\Users\\EnzoGamer\\AppData\\Local\\conda\\conda\\envs\\tf_gpu\\python.exe \"C:\\Users\\EnzoGamer\\Desktop\\PROJET IA\\cnn trainer\\CNNTrainer\\script.py\""

        t = Thread(target = lambda: leterminal(command, self.text))
        t.start()













        self.label_pythonpath = Label(self, text="Chemin vers l'interpreteur python: " + self.pythonpath)
        self.label_pythonpath.pack(side="left")

        self.bouton_cliquer = Button(self, text="Changer d'interpreteur python", fg="red", command=self.change_pyhton_path)
        self.bouton_cliquer.pack(side="right")
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
                self.label_dataset_train["text"] = str(dir_train_nb_fic) + " photos"
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
                self.label_dataset_validation["text"] = str(dir_validation_nb_fic) + " photos"
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
        else:
            self.bouton_choose_model.config(state=DISABLED)

    def change_pyhton_path(self):
        self.choose_pyhton_path()
        print("affichage de pythonpath :")
        print(self.pythonpath)
        print("modification du label de texte ..")
        self.label_pythonpath["text"] = "Chemin vers l'interpreteur python: " + self.pythonpath

    def choose_pyhton_path(self):
        temp = askopenfilename(title="Indiquer l'interpreteur python à utiliser",filetypes=[('python','.*'),('all files','.*')])
        if(os.path.isfile(temp)):
            self.pythonpath = temp
            changer_xml(balise="pyhton_path", attribut="path", valeur=temp)
        else:
            print("le chemin que vous venez de donner n'est pas valide")

    def choose_model(self):
        self.modelpath = askopenfilename(title="Choisir un modèle de réseau de neurones convolutionnel",filetypes=[('h5 files','.h5'),('all files','.*')])




fenetre = Tk()
interface = Interface(fenetre)

interface.mainloop()
interface.destroy()