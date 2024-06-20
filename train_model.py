import tensorflow as tf
from tensorflow.keras.applications import VGG16
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import layers, models
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.regularizers import l2

def create_datagen(train_dir, validation_dir):
    # Geradores de dados com aumento de dados para treinamento e validação
    train_datagen = ImageDataGenerator(
        rescale=1./255,
        rotation_range=40,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        fill_mode='nearest'
    )
    validation_datagen = ImageDataGenerator(rescale=1./255)

    train_generator = train_datagen.flow_from_directory(
        train_dir,
        target_size=(150, 150),
        batch_size=32,
        class_mode='categorical'
    )

    validation_generator = validation_datagen.flow_from_directory(
        validation_dir,
        target_size=(150, 150),
        batch_size=32,
        class_mode='categorical'
    )

    return train_generator, validation_generator

def create_model():
    # Base model com VGG16

    base_model = VGG16(weights='imagenet', include_top=False, input_shape=(150, 150, 3))
    base_model.trainable = False  # Congelar as camadas convolucionais do VGG16

    # Adicionando camadas personalizadas
    model = models.Sequential([
        base_model,
        layers.Flatten(),
        layers.Dense(512, activation='relu', kernel_regularizer=l2(0.001)),
        layers.Dropout(0.5),
        layers.Dense(8, activation='softmax')  # 8 classes
    ])

    model.compile(optimizer=Adam(lr=1e-4), loss='categorical_crossentropy', metrics=['accuracy'])
    return model

def train_model(model, train_generator, validation_generator):
    # Treinando o modelo com ajuste fino nas últimas camadas do VGG16 após algumas épocas
    history = model.fit(
        train_generator,
        steps_per_epoch=train_generator.samples // 32,
        epochs=30,
        validation_data=validation_generator,
        validation_steps=validation_generator.samples // 32
    )

    # Desbloquear algumas camadas do VGG16 para ajuste fino
    base_model = model.layers[0]
    for layer in base_model.layers[-4:]:
        layer.trainable = True

    model.compile(optimizer=Adam(lr=1e-5),  # Taxa de aprendizado menor para ajuste fino
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])

    # Continuar treinamento (ajuste fino)
    fine_tune_history = model.fit(
        train_generator,
        steps_per_epoch=train_generator.samples // 32,
        epochs=10,  # Menos épocas para ajuste fino
        validation_data=validation_generator,
        validation_steps=validation_generator.samples // 32
    )
    return history, fine_tune_history

def save_model(model, path='model/model.h5'):
    # Salvando o modelo treinado
    model.save(path)

# Diretórios das imagens
train_dir = 'dataset/train'
validation_dir = 'dataset/validation'

# Criação de geradores de dados
train_generator, validation_generator = create_datagen(train_dir, validation_dir)

# Criação do modelo
model = create_model()

# Treinamento do modelo
history, fine_tune_history = train_model(model, train_generator, validation_generator)

# Salvando o modelo
save_model(model)
