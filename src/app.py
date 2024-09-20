from utils import db_connect
engine = db_connect()

# your code here
#Librerias 
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.feature_selection import SelectKBest, f_regression
from sklearn.linear_model import Lasso
import matplotlib.pyplot as plt
import numpy as np

#Cargar csv
df=pd.read_csv("https://raw.githubusercontent.com/4GeeksAcademy/regularized-linear-regression-project-tutorial/main/demographic_health_data.csv")
pd.set_option('display.max_columns', None)
df.head()
num_rows, num_columns = df.shape
print(f"Número de filas: {num_rows}")
print(f"Número de columnas: {num_columns}")

#EDA
#Limpiar datos

nans= df.isna()
dupli= df.duplicated()
df.nunique()
nanss= df.isna().sum()
dupli.sum()

df.dropna()
df.drop_duplicates()

df["COUNTY_NAME"].nunique()
df["STATE_NAME"].nunique()
df["Heart disease_number"].nunique()

df_tipo = df.select_dtypes(include= ["int64", "float"])
df_tipo.head()

#Normalizar los datos numéricos

scaler = StandardScaler()
df_norm = pd.DataFrame(scaler.fit_transform(df_tipo), index=df_tipo.index, columns=df_tipo.columns)
df_norm.head()

#Train y test
X = df_norm.drop(columns=["Heart disease_number"])
y = df_norm["Heart disease_number"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

k = int(len(X_train.columns) * 0.3)
selec_model = SelectKBest(score_func = f_regression, k = k)
selec_model.fit(X_train, y_train)
modelo = selec_model.get_support()

X_train_sel = pd.DataFrame(selec_model.transform(X_train), columns = X_train.columns.values[modelo])
X_test_sel = pd.DataFrame(selec_model.transform(X_test), columns = X_test.columns.values[modelo])

X_train_sel["Heart disease_number"] = list(y_train)
X_test_sel["Heart disease_number"] = list(y_test)

#Falyta , index = False
X_train_sel.to_csv("../data/processed/clean_train.csv")
X_test_sel.to_csv("../data/processed/clean_test.csv") 

total_datos = pd.concat([X_train_sel, X_test_sel])
total_datos.head()

#Regresión linear

train_datos = pd.read_csv("../data/processed/clean_train.csv")
test_datos = pd.read_csv("../data/processed/clean_test.csv")

X_train = train_datos.drop(columns=["Heart disease_number"])
y_train = train_datos["Heart disease_number"]

X_test = test_datos.drop(columns=["Heart disease_number"])
y_test = test_datos["Heart disease_number"]

# Entrenar el modelo de regresión lineal
lm = LinearRegression()
lm.fit(X_train, y_train)

# Predecir con el modelo de regresión lineal
y_train_pred = lm.predict(X_train)
y_test_pred = lm.predict(X_test)

# Calcular R^2 para el modelo de regresión lineal
train_r2_linear = r2_score(y_train, y_train_pred)
test_r2_linear = r2_score(y_test, y_test_pred)

print(f"R2 train {train_r2_linear}")
print(f"R2 test {test_r2_linear}")

#Laso
#Prueba con las alphas de 0 a 20 
alphas = [0.0, 0.1, 0.5, 1.0, 5.0, 10.0, 20.0]

train_r2_laso = []
test_r2_laso = []

for alpha in alphas:
    laso_modelo = Lasso(alpha=alpha)
    laso_modelo.fit(X_train, y_train)
    
    # Predecir
    y_train_pred_laso = laso_modelo.predict(X_train)
    y_test_pred_laso = laso_modelo.predict(X_test)
    
    # Calcular R^2
    train_r2_laso.append(r2_score(y_train, y_train_pred_laso))
    test_r2_laso.append(r2_score(y_test, y_test_pred_laso))

# Graficar la evolución de R^2 con diferentes valores de alpha
plt.figure(figsize=(10,6))
plt.plot(alphas, train_r2_laso, label="Entrenamiento")
plt.plot(alphas, test_r2_laso, label="Prueba")
plt.xlabel("Alpha")
plt.ylabel("R2")
plt.legend()
plt.show()
alphas = [0. , 0.2, 0.4, 0.6, 0.8, 1. , 1.2, 1.4, 1.6, 1.8, 2. , 2.2, 2.4]

from sklearn.model_selection import GridSearchCV

# Definir un rango más amplio de valores para alpha
alpha_values = {'alpha': [0.1, 0.5, 1, 5, 10, 20]}

# GridSearchCV para Lasso
lasso = Lasso()
grid_search = GridSearchCV(lasso, param_grid=alpha_values, cv=5, scoring='r2')
grid_search.fit(X_train, y_train)

# Mejor hiperparámetro encontrado
best_alpha = grid_search.best_params_['alpha']
print(f"Mejor valor de alpha: {best_alpha}")

# Entrenar el modelo Lasso con el mejor valor de alpha
best_lasso_model = Lasso(alpha=best_alpha)
best_lasso_model.fit(X_train, y_train)

# Evaluar el modelo con el mejor alpha
y_train_best_lasso = best_lasso_model.predict(X_train)
y_test_best_lasso = best_lasso_model.predict(X_test)

train_r2_best_lasso = r2_score(y_train, y_train_best_lasso)
test_r2_best_lasso = r2_score(y_test, y_test_best_lasso)

print(f"Mejor R^2 en entrenamiento (Lasso optimizado): {train_r2_best_lasso}")
print(f"Mejor R^2 en prueba (Lasso optimizado): {test_r2_best_lasso}")