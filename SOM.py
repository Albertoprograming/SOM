import h5py
import numpy as np
import matplotlib.pyplot as plt
from minisom import MiniSom
from sklearn.preprocessing import StandardScaler


train_x_path = r"D:\Intelingencia Artificial\IA.6Semestre\cartel_exposicion\camelyonpatch_level_2_split_train_x.h5"
train_y_path = r"D:\Intelingencia Artificial\IA.6Semestre\cartel_exposicion\camelyonpatch_level_2_split_train_y.h5"

# Leer las imágenes y etiquetas
with h5py.File(train_x_path, 'r') as file_x, h5py.File(train_y_path, 'r') as file_y:
    x_train = file_x['x'][:]  # Imágenes de entrenamiento
    y_train = file_y['y'][:]  # Etiquetas de entrenamiento

# Usar solo una porción más pequeña del dataset para pruebas iniciales
small_fraction = x_train.shape[0] // 32  # Usar una porción más pequeña
x_train = x_train[:small_fraction]
y_train = y_train[:small_fraction]

# Preprocesamiento: aplanar las imágenes y normalizar los valores de píxeles
x_train_flat = x_train.reshape(x_train.shape[0], -1)  # Convertir a vectores
x_train_flat = x_train_flat.astype(np.float32)  # Convertir a float32 para ahorrar memoria

scaler = StandardScaler()

# Utilizar aprendizaje incremental en bloques más pequeños
for chunk in np.array_split(x_train_flat, 50):  # Dividir datos en 50 partes
    scaler.partial_fit(chunk)

x_train_scaled = scaler.transform(x_train_flat)  # Transformar todos los datos

# Configuración del SOM
som_size = 5  # Tamaño de la cuadrícula SOM (5x5)
som = MiniSom(som_size, som_size, x_train_scaled.shape[1], sigma=1.0, learning_rate=0.5)

# Inicializar pesos y entrenar el SOM
som.random_weights_init(x_train_scaled)
print("Entrenando el SOM...")
som.train_random(x_train_scaled, 45)  # 50 iteraciones de entrenamiento
print("Entrenamiento completado.")
def calculate_u_matrix(som):
    """
    Calcula la U-Matrix (Matriz Unificada de Distancias) para un SOM entrenado.
    Args:
        som (MiniSom): Objeto MiniSom entrenado.
    Returns:
        np.ndarray: Matriz de distancias unificadas (U-Matrix).
    """
    weights = som._weights
    x, y, _ = weights.shape
    u_matrix = np.zeros((x, y))

    for i in range(x):
        for j in range(y):
            neighbors = []
            if i > 0:
                neighbors.append(weights[i - 1, j])  # Arriba
            if i < x - 1:
                neighbors.append(weights[i + 1, j])  # Abajo
            if j > 0:
                neighbors.append(weights[i, j - 1])  # Izquierda
            if j < y - 1:
                neighbors.append(weights[i, j + 1])  # Derecha

            # Calcular la distancia promedio con los vecinos
            distances = [np.linalg.norm(weights[i, j] - neighbor) for neighbor in neighbors]
            u_matrix[i, j] = np.mean(distances)

    return u_matrix

# Calcular la U-Matrix
u_matrix = calculate_u_matrix(som)

# Visualizar la U-Matrix
plt.figure(figsize=(8, 6))
plt.title("U-Matrix (Mapa de Distancia Unificada)")
plt.imshow(u_matrix, cmap='coolwarm', interpolation='nearest')
plt.colorbar(label='Distancia')
plt.show()


# Visualización del mapa de distancia de pesos
plt.figure(figsize=(8, 6))
plt.title("Mapa de distancia (SOM)")
plt.pcolor(som.distance_map().T, cmap='coolwarm')  # Mapa de distancias
plt.colorbar(label='Distancia')
plt.show()

# Asignar las etiquetas a las neuronas ganadoras
winner_coordinates = np.array([som.winner(x) for x in x_train_scaled])  # Coordenadas del SOM
frequencies = np.zeros((som_size, som_size))

# Calcular frecuencias de cada nodo
for coord in winner_coordinates:
    frequencies[coord[0], coord[1]] += 1  # Corregir la indexación

# Visualizar la frecuencia de activaciones en el SOM
plt.figure(figsize=(8, 6))
plt.title("Frecuencia de activaciones en el SOM")
plt.pcolor(frequencies.T, cmap='Blues')
plt.colorbar(label='Frecuencia')
plt.show()

# Filtrar coordenadas de ganadores por clase
indices_class_0 = np.where(y_train == 0)[0]
indices_class_1 = np.where(y_train == 1)[0]

class_0 = winner_coordinates[indices_class_0]
class_1 = winner_coordinates[indices_class_1]

# Visualizar la distribución de clases en el SOM
plt.figure(figsize=(8, 6))
plt.title("Distribución de clases en el SOM")
plt.scatter(class_0[:, 0], class_0[:, 1], label='Clase 0 (Normal)', alpha=0.6, color='blue')
plt.scatter(class_1[:, 0], class_1[:, 1], label='Clase 1 (Patológica)', alpha=0.6, color='red')
plt.legend()
plt.show()
