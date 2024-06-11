from django.shortcuts import render
import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib import gridspec
from io import BytesIO
import base64

def index(request):
    # Cargar el archivo CSV
    df = pd.read_csv("creditcard.csv")
    
    # Información básica del DataFrame
    df_length = len(df)
    df_num_features = len(df.columns)
    class_distribution = df["Class"].value_counts().to_dict()
    missing_values = df.isna().any().to_dict()

    # Configuración de la figura del gráfico para múltiples gráficos
    plt.figure(figsize=(12, 40))  # Aumentar el tamaño para acomodar ambos gráficos
    gs = gridspec.GridSpec(9, 4)  # Ajustar para más filas
    gs.update(hspace=0.9)

    # Generar gráficos para cada característica
    for i, f in enumerate(df.drop("Class", axis=1).columns):
        if i < 32:  # Limitar el número de gráficos para evitar errores
            ax = plt.subplot(gs[i])
            sns.distplot(df[f][df["Class"] == 1], label='Class 1', hist=False)
            sns.distplot(df[f][df["Class"] == 0], label='Class 0', hist=False)
            ax.set_xlabel('')
            ax.set_title('feature: ' + str(f))
    
    # Añadir gráfico de dispersión para V10 y V14
    ax = plt.subplot(gs[32])  # Usar la próxima fila disponible en GridSpec
    plt.scatter(df["V10"][df['Class'] == 0], df["V14"][df['Class'] == 0], c="g", marker=".", label='Class 0')
    plt.scatter(df["V10"][df['Class'] == 1], df["V14"][df['Class'] == 1], c="r", marker=".", label='Class 1')
    plt.xlabel("V10", fontsize=14)
    plt.ylabel("V14", fontsize=14)
    plt.title("Scatter Plot of V10 vs V14")
    plt.legend()

    # Guardar la figura en un buffer y luego codificar en base64 para pasar al HTML
    buf = BytesIO()
    plt.savefig(buf, format='png')
    plt.close()
    image_base64 = base64.b64encode(buf.getvalue()).decode('utf-8')

    # Pasar los datos al template
    context = {
        'df_head': df.head(10).to_html(classes='table table-striped', index=False),
        'df_length': df_length,
        'df_num_features': df_num_features,
        'class_distribution': class_distribution,
        'missing_values': missing_values,
        'image_base64': image_base64,
    }
    return render(request, 'home/section17.html', context)