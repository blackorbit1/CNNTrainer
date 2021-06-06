import yaml

# PATHS
CONFIG_PATH = "config.yml"
TENSORBOARD_LOG_PATH = "./logs"
LOGO_PATH = "CNNTrainer/top_logo.png"

# GENERATED FILES
AUTO_SAVED_MODEL_NAME = "-wi-{epoch:02d}-{val_acc:.2f}.hdf5"

# VIEW SETTINGS
WINDOW_WIDTH = 400
WINDOW_HEIGHT = 1000

VIEW_DISABLED = None
VIEW_NORMAL = None

with open(CONFIG_PATH, 'r') as stream:
    CONFIG = yaml.safe_load(stream)

def CONFIG_KEYS(keys):
    tmp = CONFIG
    for key in keys:
        tmp = list(tmp[key].keys())
    return tmp
