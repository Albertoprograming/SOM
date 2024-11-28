import h5py
import matplotlib.pyplot as plt
import numpy as np
# Ruta del archivo de imágenes de validación (ajusta según tu directorio)
valid_images_file = r"D:\Intelingencia Artificial\IA.6Semestre\cartel_exposicion\camelyonpatch_level_2_split_valid_x.h5"
valid_labels_file = r"D:\Intelingencia Artificial\IA.6Semestre\cartel_exposicion\camelyonpatch_level_2_split_valid_y.h5"

try:
    # Cargar las imágenes del archivo de validación
    with h5py.File(valid_images_file, 'r') as hf:
        images = hf['x'][:]  # Dataset con las imágenes

    # Cargar las etiquetas del archivo de validación
    with h5py.File(valid_labels_file, 'r') as hf:
        labels = hf['y'][:]  # Dataset con las etiquetas

    # Contar el número de etiquetas únicas y su frecuencia
    unique_labels, counts = np.unique(labels, return_counts=True)
    label_counts = dict(zip(unique_labels, counts))
    print(f"Etiquetas únicas y sus frecuencias: {label_counts}")

    # Determinar la cantidad mínima de imágenes por etiqueta para mostrar una cantidad equitativa
    min_count = min(counts)
    num_to_show = min(10, min_count)  # Mostrar hasta 10 imágenes por etiqueta

    # Visualizar una cantidad equitativa de imágenes para cada etiqueta
    def plot_images(images, labels, unique_labels, num_to_show):
        plt.figure(figsize=(20, 10))  # Aumentar el tamaño de la figura
        for i, label in enumerate(unique_labels):
            label_indices = np.where(labels == label)[0][:num_to_show]
            for j, idx in enumerate(label_indices):
                plt.subplot(len(unique_labels), num_to_show, i * num_to_show + j + 1)
                plt.imshow(images[idx])
                plt.title(f"Etiqueta: {label}")
                plt.axis('off')
        plt.tight_layout(pad=3.0)  # Ajustar el espaciado entre subplots
        plt.show()

    # Mostrar una cantidad equitativa de imágenes para cada etiqueta
    plot_images(images, labels, unique_labels, num_to_show)

except PermissionError as e:
    print(f"Error de permiso: {e}")
    print("Verifica los permisos del archivo y asegúrate de que no esté en uso por otro programa.")
except Exception as e:
    print(f"Se produjo un error: {e}")