import tensorflow as tf
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import numpy as np
import os
import json
import logging
import matplotlib.pyplot as plt
from sklearn.metrics import precision_score, recall_score, f1_score
import shutil  # Para mover archivos

# ================================
# CONFIGURAR LOGGING
# ================================
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# ================================
# HABILITAR CUDA EN SERVIDOR
# ================================
if tf.config.list_physical_devices('GPU'):
    try:
        for gpu in tf.config.list_physical_devices('GPU'):
            tf.config.experimental.set_memory_growth(gpu, True)
        logging.info("✅ TensorFlow ejecutándose en GPU con CUDA.")
    except RuntimeError as e:
        logging.error(f"Error al configurar CUDA: {str(e)}")
else:
    logging.warning("⚠️ No se detectó GPU. Usando CPU.")

# ================================
# ACTIVA PRECISIÓN MIXTA PARA CUDA
# ================================
from tensorflow.keras import mixed_precision
mixed_precision.set_global_policy('mixed_float16')
logging.info("✅ Precisión mixta activada (float16).")

# ================================
# CONFIGURACIÓN DEL DATASET
# ================================
dataset_path = '/workspace/Dataset_train_CNN_clasificadora'  # Ruta en el servidor

IMG_SIZE = (320, 320)  # Reducido para mayor velocidad
BATCH_SIZE = 32  # Aumentado para aprovechar la GPU

datagen = ImageDataGenerator(
    rescale=1./255,
    validation_split=0.2,
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True
)

train_generator = datagen.flow_from_directory(
    os.path.join(dataset_path, "train"),
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='binary',
    subset='training'
)

val_generator = datagen.flow_from_directory(
    os.path.join(dataset_path, "val"),
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='binary',
    subset='validation'
)

logging.info("Dataset cargado correctamente.")

# ================================
# DEFINIR Y ENTRENAR LA CNN
# ================================
model = Sequential([
    tf.keras.layers.Input(shape=(320, 320, 3)),  # Definir la capa de entrada
    Conv2D(32, (3, 3), activation='relu'),
    MaxPooling2D(2, 2),

    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D(2, 2),

    Conv2D(128, (3, 3), activation='relu'),
    MaxPooling2D(2, 2),

    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(1, activation='sigmoid')
])

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

history = model.fit(
    train_generator,
    validation_data=val_generator,
    epochs=10
)

logging.info("Modelo entrenado correctamente en CUDA.")

# ================================
# GUARDAR MODELO Y CLASES
# ================================
os.makedirs("Models", exist_ok=True)

model.save("Models/modelo_cunetas.h5")

class_indices = train_generator.class_indices
with open("Models/class_indices.json", "w") as f:
    json.dump(class_indices, f)

logging.info("Modelo y clases guardadas correctamente.")

# ================================
# GRAFICAR MÉTRICAS DE ENTRENAMIENTO
# ================================
def plot_training_metrics(history):
    epochs = range(len(history.history['accuracy']))

    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.plot(epochs, history.history['accuracy'], label='Accuracy Entrenamiento')
    plt.plot(epochs, history.history['val_accuracy'], label='Accuracy Validación')
    plt.legend()
    plt.title('Accuracy durante el entrenamiento')

    plt.subplot(1, 2, 2)
    plt.plot(epochs, history.history['loss'], label='Loss Entrenamiento')
    plt.plot(epochs, history.history['val_loss'], label='Loss Validación')
    plt.legend()
    plt.title('Loss durante el entrenamiento')

    plt.show()

plot_training_metrics(history)

# ================================
# EVALUAR PRECISIÓN, RECALL Y F1-SCORE
# ================================
y_true = val_generator.classes
y_pred = model.predict(val_generator)
y_pred = (y_pred > 0.5).astype(int)

precision = precision_score(y_true, y_pred)
recall = recall_score(y_true, y_pred)
f1 = f1_score(y_true, y_pred)

logging.info(f'Precisión: {precision:.4f}')
logging.info(f'Recall: {recall:.4f}')
logging.info(f'F1-Score: {f1:.4f}')

# ================================
# CREAR CARPETAS PARA CLASIFICACIÓN
# ================================
output_cunetas = os.path.join("Imagenes_CNN_seleccionadas", "cuneta")
output_no_cunetas = os.path.join("Imagenes_CNN_seleccionadas", "no_cuneta")

os.makedirs(output_cunetas, exist_ok=True)
os.makedirs(output_no_cunetas, exist_ok=True)

def clasificar_imagen(ruta_imagen):
    img = image.load_img(ruta_imagen, target_size=IMG_SIZE)
    img_array = image.img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    prediccion = model.predict(img_array)[0][0]
    clase_predicha = "cuneta" if prediccion > 0.5 else "no_cuneta"

    destino_dir = output_cunetas if clase_predicha == "cuneta" else output_no_cunetas
    destino = os.path.join(destino_dir, os.path.basename(ruta_imagen))

    shutil.move(ruta_imagen, destino)

    logging.info(f"Imagen {ruta_imagen} clasificada como {clase_predicha} y movida a {destino}")
    return f"{ruta_imagen} → {clase_predicha}"

carpeta_imagenes = "Prueba_CNN_segmentadas/"
for archivo in os.listdir(carpeta_imagenes):
    ruta_completa = os.path.join(carpeta_imagenes, archivo)
    if archivo.lower().endswith(('.png', '.jpg', '.jpeg')):
        print(clasificar_imagen(ruta_completa))
