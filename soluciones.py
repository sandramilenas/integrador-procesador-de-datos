

from typing import Tuple
import matplotlib.pyplot as plt
import numpy as np
from numpy.random import default_rng
from sklearn.datasets import load_digits
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split


def ej_1_carga_dataset() -> Tuple[np.ndarray, np.ndarray]:
    """Cargar los dígitos y retornar las matrices de imágenes y etiquetas
    Returns:
        Tuple[np.ndarray, np.ndarray]: Tuple[np.array[shape=(n_imgs, 8, 8)], np.array[shape=(n_imgs,)]]
    """
    # Cargar el dataset
    digits = load_digits()

    # Obtener las imágenes y etiquetas
    images = digits.images
    labels = digits.target

    # Retornar las matrices de imágenes y etiquetas
    return images, labels



def ej_2_exploracion_dataset(X: np.ndarray, Y: np.ndarray, rng: np.random.Generator):
    """Graficar 10 imágenes
    Esta función debe utilizar el generador de números aleatorios `rng` para selccionar
    10 números al azar en el rango (0, len(X)), para esto usa la función `rng.choice`,
    la cual retorna un array de índices que puedes usar para selccionar la imágenes y
    etiquetas a graficar

    Grafica cada imágen con `plt.imshow` y configura el título de cada gráfica de
    la forma `Digito: <etiqueta>`

    Args:
        X (np.ndarray): Imágenes
        Y (np.ndarray): Etiquetas
        rng (np.random.Generator): Generador de números aleatorios
    """


# Selecciona 10 números al azar
    random_indices = rng.choice(np.array(range(len(X))),size=10,replace=False)
    print(random_indices,X.shape,Y.shape)


    # Graficar las imágenes seleccionadas
    plt.figure(figsize=(12, 6))
    for i, index in enumerate(random_indices):
        plt.imshow(X[index], cmap='gray')
        plt.title(f"Digito: {Y[index]}")
        plt.axis('off')
        plt.tight_layout()
        plt.show()




def ej_3_entrenar_arbol(X: np.ndarray, y: np.ndarray) -> DecisionTreeClassifier:
    """Entrenar un árbol de decisiones y retornarlo como resultado de la función
    No olvides convertir la matriz X a las dimensiones correctas

    Args:
        X (np.ndarray): Matriz de dimensiones (n_imagenes, 8, 8)
        y (np.ndarray): Matriz de dimensiones (n_imagenes,)

    Returns:
        DecisionTreeClassifier: Modelo ajustado
    """
    digits = load_digits()

    # Aplana las imágenes a un vector
    X = digits.data.reshape(-1, 64)

    # Divide los datos en conjuntos de entrenamiento y prueba
    X_train, X_test, y_train, y_test = train_test_split(X, digits.target, test_size=0.2, random_state=42)

    # Crea una instancia del clasificador de árbol de decisión
    clf = DecisionTreeClassifier(random_state=42)

    # Entrena el modelo
    clf.fit(X_train, y_train)
    return clf


def ej_4_evaluacion_rendimiento(
    modelo: DecisionTreeClassifier, X_test: np.ndarray, y_test: np.ndarray
) -> float:
    """Calcula el accuracy del modelo y retornalo como resultado de la función

    Args:
        modelo (DecisionTreeClassifier): Modelo ya entrenado
        X_test (np.ndarray): Matriz con datos de test de dimensión (n_imagenes, 64)
        y_test (np.ndarray): Matriz con etiquetas de test de dimensión (n_imagenes,)

    Returns:
        float: accuracy calculada

    """
    digits = load_digits()

    # Aplana las imágenes a un vector
    X = digits.data.reshape(-1, 64)

    # Divide los datos en conjuntos de entrenamiento y prueba
    X_train, X_test, y_train, y_test = train_test_split(X, digits.target, test_size=0.2, random_state=42)

    # Crea una instancia del clasificador de árbol de decisión
    clf = DecisionTreeClassifier(random_state=42)

    # Entrena el modelo
    clf.fit(X_train, y_train)
    score = clf.score(X_test, y_test)
    return score