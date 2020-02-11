# CNNTrainer

![logo du logiciel](https://github.com/blackorbit1/CNNTrainer/blob/master/icone_cnn_trainer_mini.png?raw=true)

Logiciel pour faciliter l'entrainement d'un réseau de neurones convolutionnel

![capture du logiciel](https://github.com/blackorbit1/CNNTrainer/blob/master/capture_cnntrainer.JPG?raw=true)


---> Attention, pour executer ce logiciel, tkinter doit etre installé sur python


### Utilité du pré-entrainement
Lorsque vous faites du fine-tuning partiel sur un modèle de base déjà entrainé, CNNTrainer retire la derniere couche contenant généralement 1000 classes pour en mettre une nouvelle avec le nombre de classes désiré.
Or, en faisant cela, les poids synaptiques sont mis à 0, on fait donc souvent un pré-entrainement pour redonner des valeurs cohérentes aux poids synaptiques de cette dernière couche.

Cette étape n'est cependant pas obligatoire.


### Interpréteur Python
Pour utiliser CNNTrainer, vous devez lui indiquer l'interpréteur Python (python.exe) à utiliser lors de son démarrage.
Sur cet interpréteur, TensorFlow et Keras doivent être installés.


### Option "Faire un fine tuning partiel"
Option à activer si vous voulez faire du fine tuning partiel en vous basant sur un fichier contenant un modèle ayant un nombre de classes différent que le nombre de classes que vous voulez pour le modèle final.
