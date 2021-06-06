import tkinter

class Logger(object):
    def __init__(self, terminal):
        #self.printer = sys.stdout
        self.terminal = terminal
        self.train_progress = None
        #self.log = open("logfile.log", "a")
    def write(self, message):
        #self.printer.write(message)
        self.terminal.insert(tkinter.END, message)
        self.terminal.see(tkinter.END)

        if self.train_progress is not None:
            if "Lancement de l'entrainement ..." in message:
                self.train_progress["value"] = 5
            try:
                if "Epoch " in message:
                    avancement = message[6:]
                    #print(avancement)
                    actuel, final = avancement.split("/")
                    #print("actuel : " + actuel + " / final : " + final)
                    final_amount = int(actuel) * 100 / int(final)
                    #print(int(final_amount))
                    self.train_progress["value"] = final_amount
            except ValueError:
                pass
    def set_training_bar(self, training_bar):
        self.train_progress = training_bar
    def flush(self):
        #this flush method is needed for python 3 compatibility.
        #this handles the flush command by doing nothing.
        #you might want to specify some extra behavior here.
        pass
