from django.shortcuts import redirect
from django.shortcuts import render
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
from sklearn.ensemble import RandomForestClassifier

# Función para dividir el conjunto de datos en entrenamiento, validación y prueba
def train_val_test_split(df, rstate=42, shuffle=True, stratify=None):
    strat = df[stratify] if stratify else None
    train_set, test_set = train_test_split(
        df, test_size=0.4, random_state=rstate, shuffle=shuffle, stratify=strat)
    strat = test_set[stratify] if stratify else None
    val_set, test_set = train_test_split(
        test_set, test_size=0.5, random_state=rstate, shuffle=shuffle, stratify=strat)
    return (train_set, val_set, test_set)

# Función para quitar las etiquetas del conjunto de datos
def remove_labels(df, label_name):
    X = df.drop(label_name, axis=1)
    y = df[label_name].copy()
    return (X, y)

# Función para calcular el F1 score después de reducir características
def calcular_f1_score_reduced(y_true, y_pred):
    f1_score_reduced = f1_score(y_true, y_pred, average='weighted')
    return f1_score_reduced

def section14(request):
    # Lectura del conjunto de datos
    df = pd.read_csv('TotalFeatures-ISCXFlowMeter.csv')

    # División del conjunto de datos
    train_set, val_set, test_set = train_val_test_split(df)

    X_train, y_train = remove_labels(train_set, 'calss')
    X_val, y_val = remove_labels(val_set, 'calss')
    X_test, y_test = remove_labels(test_set, 'calss')

    # Modelo Random Forest
    clf_rnd = RandomForestClassifier(n_estimators=50, random_state=42, n_jobs=-1)
    clf_rnd.fit(X_train, y_train)

    # Predicciones con el conjunto de datos de validación
    y_pred = clf_rnd.predict(X_val)

    # F1 score en el conjunto de datos de validación
    f1_score_val = f1_score(y_val, y_pred, average='weighted')

    # Características más importantes
    feature_importances = {name: score for name, score in zip(list(df), clf_rnd.feature_importances_)}
    feature_importances_sorted = pd.Series(feature_importances).sort_values(ascending=False)
    columns = list(feature_importances_sorted.head(10).index)
    X_train_reduced = X_train[columns].copy()
    X_val_reduced = X_val[columns].copy()

    # Modelo Random Forest con características reducidas
    clf_rnd_reduced = RandomForestClassifier(n_estimators=50, random_state=42, n_jobs=-1)
    clf_rnd_reduced.fit(X_train_reduced, y_train)

    # Predicciones con el conjunto de datos de validación con características reducidas
    y_pred_reduced = clf_rnd_reduced.predict(X_val_reduced)

    # F1 score en el conjunto de datos de validación con características reducidas
    f1_score_val_reduced = calcular_f1_score_reduced(y_val, y_pred_reduced)

    context = {
        'f1_score_val': f1_score_val,
        'f1_score_val_reduced': f1_score_val_reduced,
        'df': df,
        'feature_importances_sorted': feature_importances_sorted,
        'columns': columns,
        'X_train_reduced': X_train_reduced,
        # Otros datos de contexto aquí...
    }

    return render(request, 'home/section14.html', context)

def index(request):
    # Redirigir a la vista section14
    return render(request, 'home/index.html')
