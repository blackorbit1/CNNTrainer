def get_optimiser(pas, optimiser):
    if optimiser == "RMSprop":
        from tensorflow.keras.optimizers import RMSprop
        return RMSprop(lr=pas, rho=0.9, epsilon=None, decay=0.0)
    elif optimiser == "AMSGrad":
        from tensorflow.keras.optimizers import Adam
        return Adam(lr=pas, beta_1=0.9, beta_2=0.999, epsilon=1e-8, amsgrad=True)
    elif optimiser == "Adam":
        from tensorflow.keras.optimizers import Adam
        return Adam(lr=pas, beta_1=0.9, beta_2=0.999, epsilon=1e-8, amsgrad=False)
    elif optimiser == "SGD":
        from tensorflow.keras.optimizers import SGD
        return SGD(lr=pas, momentum=0.0, decay=0.0, nesterov=False)
    else:
        from tensorflow.keras.optimizers import RMSprop
        return RMSprop(lr=pas, rho=0.9, epsilon=None, decay=0.0)

def get_model(model_name, width=224, height=224, nb_layers=3, pretraining_dataset="imagenet"):
    if model_name == "ResNet50":
        from tensorflow.keras.applications.resnet import ResNet50
        return ResNet50(weights=pretraining_dataset, include_top=False, input_shape=(width, height, nb_layers))
    elif model_name == "InceptionResNetV2":
        from tensorflow.keras.applications.inception_resnet_v2 import InceptionResNetV2
        return InceptionResNetV2(weights=pretraining_dataset, include_top=False, input_shape=(width, height, nb_layers))
    elif model_name == "NASNetMobile":
        from tensorflow.keras.applications.nasnet import NASNetMobile
        return NASNetMobile(weights=pretraining_dataset, include_top=False, input_shape=(width, height, nb_layers))
    elif model_name == "InceptionV3":
        from tensorflow.keras.applications.inception_v3 import InceptionV3
        return InceptionV3(weights=pretraining_dataset, include_top=False, input_shape=(width, height, nb_layers))


def add_new_last_layer(model, nb_classes):
    from keras.models import Model
    from keras.layers import Dense, GlobalAveragePooling2D

    # ajout d'une couche qui va faire le lien entre l'ancienne sortie et la nouvelle sortie
    x = model.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(units=nb_classes, activation='relu')(x)

    # ajout d'une nouvelle couche de sorties
    predictions = Dense(int(nb_classes), activation='softmax')(x)

    return Model(input=model.input, output=predictions)

def use_gpu():
    from tensorflow.python.client import device_lib
    from tensorflow.keras import backend
    import tensorflow as tf

    config = tf.ConfigProto(
        gpu_options=tf.GPUOptions(force_gpu_compatible=True)
        # device_count = {'GPU': 1}
    )
    config.gpu_options.allow_growth = True
    session = tf.Session(config=config)
    backend.set_session(session)
    tf.ConfigProto().gpu_options.allow_growth = True

    print(device_lib.list_local_devices())

    # confirm TensorFlow sees the GPU
    from tensorflow.python.client import device_lib
    assert 'GPU' in str(device_lib.list_local_devices())

    # confirm Keras sees the GPU
    assert len(backend.tensorflow_backend._get_available_gpus()) > 0
    import time
    time.sleep(1)