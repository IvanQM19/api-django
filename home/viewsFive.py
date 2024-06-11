import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import matplotlib.gridspec as gridspec
from collections import Counter
from sklearn import metrics
import numpy as np
from django.shortcuts import render
import base64
from io import StringIO
import io  # Importa el módulo io

def purity_score(y_true, y_pred):
    # compute contingency matrix (also called confusion matrix)
    contingency_matrix = metrics.cluster.contingency_matrix(y_true, y_pred)
    # return purity
    return np.sum(np.amax(contingency_matrix, axis=0)) / np.sum(contingency_matrix)

def section18(request):
    # Lectura del conjunto de datos
    df = pd.read_csv("creditcard.csv")
    
    # Visualización de las primeras 10 filas del conjunto de datos
    df_head = df.head(10)
    
    info_output = StringIO()
    df.info(buf=info_output)
    info_output.seek(0)
    info_text = info_output.read()
    
    # Obtener los datos como listas de Python
    rows = df_head.values.tolist()  # Convertir DataFrame a lista de listas
    
    # Obtener los nombres de las columnas
    columns = df_head.columns.tolist()
    
    # Obtener el número de características y la longitud del conjunto de datos
    num_features = len(df.columns)
    dataset_length = len(df)
    
    # Conversión de gráficos a formato base64
    feature_plot = get_base64_encoded_image(df)
    
    context = {
        'columns': columns,
        'rows': rows,
        'info_text':info_text,
        'num_features': num_features,
        'dataset_length': dataset_length,
        'df_head': df_head,
        'feature_plot': feature_plot,
    }
    
    return render(request, 'home/section18.html', context)

def get_base64_encoded_image(df):
    # Representación gráfica de las características
    features = df.drop("Class", axis=1)
    plt.figure(figsize=(12, 32))
    gs = gridspec.GridSpec(8, 4)
    gs.update(hspace=0.8)
    for i, f in enumerate(features):
        ax = plt.subplot(gs[i])
        sns.distplot(df[f][df["Class"] == 1])
        sns.distplot(df[f][df["Class"] == 0])
        ax.set_xlabel('')
        ax.set_title('feature: ' + str(f))
    buffer = io.BytesIO()  # Usa io.BytesIO() en lugar de BytesIO()
    plt.savefig(buffer, format='png')
    plt.close()  # Cierra el gráfico después de guardarlo
    buffer.seek(0)
    img_str = base64.b64encode(buffer.read()).decode('utf-8')
    return img_str