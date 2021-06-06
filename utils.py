
def isfloat(value):
  try:
    float(value)
    return True
  except ValueError:
    return False


def change_path(path):
    content = []
    with open(__file__, "r", encoding="ISO-8859-1") as f:
        for line in f:
            content.append(line)

    with open(__file__, "w", encoding="ISO-8859-1") as f:
        content[0] = "#!{n}\n".format(n=path)
        content[1] = "path = \"{n}\"\n".format(n=path)
        for i in range(len(content)):
            f.write(content[i])

def generate_model_filename():
    import os

    dir_modele = "model_"
    nb_modele_dir = 1
    while os.path.isfile(dir_modele + str(nb_modele_dir) + ".h5"):
        nb_modele_dir += 1
    return dir_modele + str(nb_modele_dir) + ".h5"

def leterminal(command, terminal, processing_bar, bouton_lancer_entrainement):
    import sys, subprocess, tkinter
    from logger import Logger
    from global_variables import VIEW_DISABLED, VIEW_NORMAL

    original = sys.stdout
    sys.stdout = Logger(terminal)

    print("test de la fonction print")

    running = True
    bouton_lancer_entrainement.config(state=VIEW_DISABLED)
    processing_bar["value"] = 1
    p = subprocess.Popen(command, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                         universal_newlines=True)  # , env={'LANGUAGE':'en_US.en', 'LC_ALL':'en_US.UTF-8'}
    p.poll()
    processing_bar["value"] = 3

    terminal.insert(tkinter.END, "> Lancement de l'execution du script d'entrainement")
    terminal.see(tkinter.END)

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
            # print(avancement)
            actuel, final = avancement.split("/")
            # print("actuel : " + actuel + " / final : " + final)
            final_amount = int(actuel) * 100 / int(final)
            # print(int(final_amount))
            processing_bar["value"] = final_amount

        terminal.insert(tkinter.END, line)
        terminal.see(tkinter.END)
        if not line and p.poll is not None: break

    """
    while running:
        err = p.stderr.readline()
        terminal.insert(tk.END, err)
        terminal.see(tk.END)
        if not err and p.poll is not None: break
    """

    processing_bar["value"] = 0
    bouton_lancer_entrainement.config(state=VIEW_NORMAL)
    terminal.insert(tkinter.END, "-")

    sys.stdout = original