import argparse
import pandas as pd
import requests

def procesar_data_frame(df):
    if df.isnull().values.any():
        df.dropna(inplace=True)

    # Verificar filas duplicadas
    if df.duplicated().any():
        df.drop_duplicates(inplace=True)

    # Verificación y eliminación de valores atípicos
    numeric_columns = ['age', 'creatinine_phosphokinase', 'ejection_fraction', 'platelets', 'serum_creatinine','serum_sodium', 'time']

    for column in numeric_columns:
        Q1 = df[column].quantile(0.25)
        Q3 = df[column].quantile(0.75)
        IQR = Q3 - Q1

        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR

        df= df[(df[column] >= lower_bound) & (df[column] <= upper_bound)]

    df['categoria_edad'] = pd.cut(df['age'], bins=[0, 12, 19, 39, 59, float('inf')],
    labels=['Niño', 'Adolescente', 'Jóvenes adulto', 'Adulto', 'Adulto mayor'])

    df.to_csv('datos_procesados.csv', index=False)
    print("Se guardó el DataFrame procesado como datos_procesados.csv")

def main(url):
    response = requests.get(url)
    if response.status_code == 200:
        with open('heart_failure_dataset.csv', 'w') as archivo:
            archivo.write(response.text)
            print(f"El archivo heart_failure_dataset.csv se ha descargado y guardado con éxito.")
        data_frame = pd.read_csv('heart_failure_dataset.csv')
        procesar_data_frame(data_frame)
    else:
        print(f"No se pudo descargar el archivo de la URL: {url}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Script para descargar y procesar datos desde una URL.')
    parser.add_argument('url', type=str, help='URL desde donde se descargan los datos')
    args = parser.parse_args()
    main(args.url)

 #python procesar_datos.py https://huggingface.co/datasets/mstz/heart_failure/raw/main/heart_failure_clinical_records_dataset.csv