from django.shortcuts import render
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV
from sklearn.preprocessing import RobustScaler
from sklearn.metrics import f1_score
from sklearn.ensemble import RandomForestClassifier
from scipy.stats import randint

# Construcción de una función que realice el particionado completo
def train_val_test_split(df, rstate=42, shuffle=True, stratify=None):
    strat = df[stratify] if stratify else None
    train_set, test_set = train_test_split(
        df, test_size=0.4, random_state=rstate, shuffle=shuffle, stratify=strat)
    strat = test_set[stratify] if stratify else None
    val_set, test_set = train_test_split(
        test_set, test_size=0.5, random_state=rstate, shuffle=shuffle, stratify=strat)
    return (train_set, val_set, test_set)

def remove_labels(df, label_name):
    X = df.drop(label_name, axis=1)
    y = df[label_name].copy()
    return (X, y)

def section16(request):
    # Código para obtener los datos para la sección 16 desde viewsThree.py
    df = pd.read_csv('TotalFeatures-ISCXFlowMeter.csv', nrows = 10000)

    # Dividimos el conjunto de datos
    train_set, val_set, test_set = train_val_test_split(df)

    X_train, y_train = remove_labels(train_set, 'calss')
    X_val, y_val = remove_labels(val_set, 'calss')
    X_test, y_test = remove_labels(test_set, 'calss')

    # Modelo entrenado con el conjunto de datos sin escalar
    clf_rnd = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
    clf_rnd.fit(X_train, y_train)

    # Predecimos con el conjunto de datos de validación
    y_pred = clf_rnd.predict(X_val)

    # F1 Score en el conjunto de datos de validación
    f1_score_val = f1_score(y_pred, y_val, average='weighted')

    # Uso de Grid Search para selección del modelo
    param_grid = [
        {'n_estimators': [100, 500, 1000], 'max_leaf_nodes': [16, 24, 36]},
        {'bootstrap': [False], 'n_estimators': [100, 500], 'max_features': [2, 3, 4]},
    ]

    rnd_clf = RandomForestClassifier(n_jobs=-1, random_state=42)

    grid_search = GridSearchCV(rnd_clf, param_grid, cv=5, scoring='f1_weighted', return_train_score=True)
    grid_search.fit(X_train, y_train)

    best_params_grid = grid_search.best_params_
    best_estimator_grid = grid_search.best_estimator_

    cvres_grid = grid_search.cv_results_
    cv_params_grid = cvres_grid['params']
    cv_mean_test_scores_grid = cvres_grid['mean_test_score']

    print("cv_params_grid:", cv_params_grid)
    print("cv_mean_test_scores_grid:", cv_mean_test_scores_grid)

    # Uso de Randomized Search para selección del modelo
    param_distribs = {
            'n_estimators': randint(low=1, high=200),
            'max_depth': randint(low=8, high=50),
        }

    rnd_clf = RandomForestClassifier(n_jobs=-1)

    rnd_search = RandomizedSearchCV(rnd_clf, param_distributions=param_distribs,
                                    n_iter=5, cv=2, scoring='f1_weighted')
    rnd_search.fit(X_train, y_train)

    best_params_rnd = rnd_search.best_params_
    best_estimator_rnd = rnd_search.best_estimator_

    cvres_rnd = rnd_search.cv_results_
    cv_params_rnd = cvres_rnd['params']
    cv_mean_test_scores_rnd = cvres_rnd['mean_test_score']

    print("cv_params_rnd:", cv_params_rnd)
    print("cv_mean_test_scores_rnd:", cv_mean_test_scores_rnd)

    # Selección del mejor modelo
    clf_rnd = rnd_search.best_estimator_

    # Predecimos con el conjunto de datos de entrenamiento
    y_train_pred = clf_rnd.predict(X_train)

    # Predicción con el conjunto de datos de entrenamiento
    f1_score_train = f1_score(y_train_pred, y_train, average='weighted')

    # Predecimos con el conjunto de datos de validación
    y_val_pred = clf_rnd.predict(X_val)

    # Predicción con el conjunto de datos de validación
    f1_score_val = f1_score(y_val_pred, y_val, average='weighted')

    context = {
        'X_train': X_train,
        'y_train': y_train,
        'X_val': X_val,
        'y_val': y_val,
        'X_test': X_test,
        'y_test': y_test,
        'best_params_grid': best_params_grid,
        'best_estimator_grid': best_estimator_grid,
        'cv_results_grid': {
            'mean_test_score': cv_mean_test_scores_grid,
            'params': cv_params_grid,
        },
        'best_params_rnd': best_params_rnd,
        'best_estimator_rnd': best_estimator_rnd,
        'cv_results_rnd': {
            'mean_test_score': cv_mean_test_scores_rnd,
            'params': cv_params_rnd,
        },
        'f1_score_train': f1_score_train,
        'f1_score_val': f1_score_val,
    }

    return render(request, 'home/section16.html', context)