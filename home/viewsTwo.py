from django.shortcuts import render
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from matplotlib.colors import ListedColormap
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import io
import urllib
import base64

# Esta función realiza el particionado completo del conjunto de datos
def train_val_test_split(df, rstate=42, shuffle=True, stratify=None):
    strat = df[stratify] if stratify else None
    train_set, test_set = train_test_split(
        df, test_size=0.4, random_state=rstate, shuffle=shuffle, stratify=strat)
    strat = test_set[stratify] if stratify else None
    val_set, test_set = train_test_split(
        test_set, test_size=0.5, random_state=rstate, shuffle=shuffle, stratify=strat)
    return (train_set, val_set, test_set)

# Esta función elimina las etiquetas de las características del DataFrame
def remove_labels(df, label_name):
    X = df.drop(label_name, axis=1)
    y = df[label_name].copy()
    return (X, y)

def section15(request):
    # Carga el conjunto de datos
    df = pd.read_csv('TotalFeatures-ISCXFlowMeter.csv', nrows=8000)

    # Realiza el particionado completo
    train_set, val_set, test_set = train_val_test_split(df)

    # Elimina las etiquetas de las características
    X_df, y_df = remove_labels(df, 'calss')
    y_df = y_df.factorize()[0]

    # Reducción del conjunto de datos a 2 dimensiones utilizando PCA
    pca = PCA(n_components=2)
    df_reduced = pca.fit_transform(X_df)
    df_reduced = pd.DataFrame(df_reduced, columns=["c1", "c2"])

    # Gráfico de dispersión de los datos reducidos
    scatter_plot = io.BytesIO()
    plt.figure(figsize=(12, 6))
    plt.plot(df_reduced["c1"][y_df==0], df_reduced["c2"][y_df==0], "yo", label="normal")
    plt.plot(df_reduced["c1"][y_df==1], df_reduced["c2"][y_df==1], "bs", label="adware")
    plt.plot(df_reduced["c1"][y_df==2], df_reduced["c2"][y_df==2], "g^", label="malware")
    plt.xlabel("c1", fontsize=15)
    plt.ylabel("c2", fontsize=15, rotation=0)
    plt.savefig(scatter_plot, format='png')
    plt.close()

    scatter_plot.seek(0)
    scatter_plot_url = base64.b64encode(scatter_plot.read()).decode('utf-8')

    # Generamos un modelo con el conjunto de datos reducido
    clf_tree_reduced = DecisionTreeClassifier(max_depth=3, random_state=42)
    clf_tree_reduced.fit(df_reduced, y_df)

    # Representamos el límite de decisión generado por el modelo
    decision_boundary_plot = io.BytesIO()
    plt.figure(figsize=(12, 6))
    plot_decision_boundary(clf_tree_reduced, df_reduced.values, y_df)
    plt.savefig(decision_boundary_plot, format='png')
    plt.close()

    decision_boundary_plot.seek(0)
    decision_boundary_plot_url = base64.b64encode(decision_boundary_plot.read()).decode('utf-8')

    # Calculamos el F1 score para el conjunto de datos reducido
    y_pred_reduced = clf_tree_reduced.predict(df_reduced)
    f1_score_reduced = f1_score(y_df, y_pred_reduced, average='weighted')

    # Otros datos y métricas
    info_text = "Some info text..."
    plot_base64 = "..."
    f1_val_score = 0.85
    f1_test_score = 0.82
    explained_variance_ratio_formatted = "0.90"

    # Convertimos el DataFrame df_reduced a una lista de diccionarios para pasarlo como contexto
    reduced_data = df_reduced.to_dict(orient='records')

# Calculamos la proporción de varianza que se ha preservado del conjunto original
    explained_variance_ratio = pca.explained_variance_ratio_
    explained_variance_ratio_formatted = [f"{ratio:.6f}" for ratio in explained_variance_ratio]

    context = {
        'scatter_plot': scatter_plot_url,
        'decision_boundary_plot': decision_boundary_plot_url,
        'reduced_data': reduced_data,
        'info_textx': info_text,
        'plot_base64': plot_base64,
        'f1_val_score': f1_val_score,
        'f1_test_score': f1_test_score,
        'explained_variance_ratio': explained_variance_ratio_formatted,
        'f1_score_reduced': f1_score_reduced,
        # Otros datos de contexto aquí...
    }


    return render(request, 'home/section15.html', context)

# Esta función traza el límite de decisión generado por el modelo
def plot_decision_boundary(clf, X, y, plot_training=True, resolution=1000):
    mins = X.min(axis=0) - 1
    maxs = X.max(axis=0) + 1
    x1, x2 = np.meshgrid(np.linspace(mins[0], maxs[0], resolution),
                         np.linspace(mins[1], maxs[1], resolution))
    X_new = np.c_[x1.ravel(), x2.ravel()]
    y_pred = clf.predict(X_new).reshape(x1.shape)
    custom_cmap = ListedColormap(['#fafab0','#9898ff','#a0faa0'])
    plt.contourf(x1, x2, y_pred, alpha=0.3, cmap=custom_cmap)
    custom_cmap2 = ListedColormap(['#7d7d58','#4c4c7f','#507d50'])
    plt.contour(x1, x2, y_pred, cmap=custom_cmap2, alpha=0.8)
    if plot_training:
        plt.plot(X[:, 0][y==0], X[:, 1][y==0], "yo", label="normal")
        plt.plot(X[:, 0][y==1], X[:, 1][y==1], "bs", label="adware")
        plt.plot(X[:, 0][y==2], X[:, 1][y==2], "g^", label="malware")
        plt.axis([mins[0], maxs[0], mins[1], maxs[1]])               
    plt.xlabel(r"$x_1$", fontsize=18)
    plt.ylabel(r"$x_2$", fontsize=18, rotation=0)