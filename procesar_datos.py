import argparse
import pandas as pd
import requests

def procesar_data_frame(df):
    # Verificar valores faltantes
    if df.isnull().values.any():
        df = df.dropna()
        print("Se eliminaron los valores faltantes.")

    # Verificar filas duplicadas
    if df.duplicated().any():
        df = df.drop_duplicates()
        print("Se eliminaron las filas duplicadas.")

    # Verificar y eliminar valores atípicos
    Q1 = df['age'].quantile(0.25)
    Q3 = df['age'].quantile(0.75)
    IQR = Q3 - Q1
    df = df[~((df['age'] < (Q1 - 1.5 * IQR)) | (df['age'] > (Q3 + 1.5 * IQR)))]

    # Crear la columna de categoría de edades
    df['categoria_edad'] = pd.cut(df['age'], bins=[0, 12, 19, 39, 59, float('inf')],
                                  labels=['Niño', 'Adolescente', 'Jóvenes adulto', 'Adulto', 'Adulto mayor'])

    # Guardar el DataFrame procesado como csv
    df.to_csv('datos_procesados1.csv', index=False)
    print("Se guardó el DataFrame procesado como datos_procesado1.csv")

def main(url):
    response = requests.get(url)
    if response.status_code == 200:
        with open('datos_descargados.csv', 'w') as archivo:
            archivo.write(response.text)
            print(f"El archivo datos_descargados.csv se ha descargado y guardado con éxito.")
        data_frame = pd.read_csv('datos_descargados.csv')
        procesar_data_frame(data_frame)
    else:
        print(f"No se pudo descargar el archivo de la URL: {url}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Script para descargar y procesar datos desde una URL.')
    parser.add_argument('url', type=str, help='URL desde donde se descargan los datos')
    args = parser.parse_args()
    main(args.url)