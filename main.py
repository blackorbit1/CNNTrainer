
"""
Auteur : DUTRA Enzo (2021)
"""

import subprocess, os

dependencies = [
    "lxml",
    "ntpath",
    "matplotlib",
    "tensorflow",
    "keras",
    "numpy",
    "pillow",
    "webbrowser",
    "traceback",
    "trace",
    "threading",
    "yaml"
]

linux_packages = {
    "python3-tk": "tkinter"
}
macos_packages = {
    "python-tk@3.9": "tkinter"
}
windows_packages = {
    "tkinter": "tkinter"
}  # url site si winget ne le contient pas


def import_or_install_os_package(package, py_import):
    try:
        __import__(py_import)
    except (ImportError, ModuleNotFoundError):
        if os.name == "posix":  # Linux
            command = "sudo apt-get install -y " + package
            subprocess.check_call(command.split(" "))
        elif os.name == "darwin":  # MacOS
            command = "brew install " + package
            try:
                subprocess.check_call(command.split(" "))
            except:
                install_brew = '/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"'
                subprocess.check_call(install_brew.split(" "))
                subprocess.check_call(command.split(" "))
        elif os.name == "windows":  # windows
            command = "winget install " + package

            try:
                subprocess.check_call(command.split(" "))
            except:
                # webbrowser doit etre importé à ce stade là
                webbrowser.open(package)
        import_or_install_python_package(py_import)


def import_or_install_python_package(package):
    try:
        __import__(package)
    except (ImportError, ModuleNotFoundError):
        print("Le module " + package + " n'est pas installé, installation ...")
        try:
            subprocess.check_call(["python3", '-m', 'pip', 'install', package])  # install pkg
        except:
            print(
                "Les modules manquants n'ont pas pu etre installés, veuillez vous assurer que tous les modules suivants sont bien présents:")
            print(dependencies)


# installation des dépendances python
for package in dependencies: import_or_install_python_package(package)

# installtion des packages requis
os_packages = {}
if os.name == "posix": os_packages = linux_packages  # Linux
if os.name == "darwin": os_packages = macos_packages  # MacOS
if os.name == "windows": os_packages = windows_packages  # Windows
for package, py_import in linux_packages.items(): import_or_install_os_package(package, py_import)

# Lancement de la vue

from tkinter import Tk
from view import Interface

fenetre = Tk()
fenetre.title("CNNTrainer")
interface = Interface(fenetre)

interface.mainloop()
interface.destroy()
