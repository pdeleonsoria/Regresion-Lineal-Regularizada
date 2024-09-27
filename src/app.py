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

#TRAIN Y TEST
X = df_tipo.drop(columns=["Heart disease_number"])
y = df_tipo["Heart disease_number"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
X_train.head()

X_test.head()

#Normalizar los datos numéricos

scaler = StandardScaler()
X_train_esc = scaler.fit_transform(X_train)
X_test_esc = scaler.transform(X_test)

X_train_esc= pd.DataFrame(X_train_esc, index= X_train.index, columns=X_train.columns)
X_train_esc.head()

X_test_esc= pd.DataFrame(X_test_esc, index= X_test.index, columns=X_test.columns)
X_test_esc.head()

# Entrenar el modelo de regresión lineal
lm = LinearRegression()
lm.fit(X_train_esc, y_train)


# Predecir con el modelo de regresión lineal
y_train_pred = lm.predict(X_train_esc)
y_test_pred = lm.predict(X_test_esc)

# Calcular R2 para el modelo de regresión lineal
train_r2_linear = r2_score(y_train, y_train_pred)
test_r2_linear = r2_score(y_test, y_test_pred)

print(f"R2 train {train_r2_linear}")
print(f"R2 test {test_r2_linear}")

#Metricas Modelo Lineal
train_median_ae_linear = median_absolute_error(y_train, y_train_pred)
test_median_ae_linear = median_absolute_error(y_test, y_test_pred)

train_mape_linear = mean_absolute_percentage_error(y_train, y_train_pred) * 100
test_mape_linear = mean_absolute_percentage_error(y_test, y_test_pred) * 100

df_metrics_linear = pd.DataFrame({
    'R2': [train_r2_linear, test_r2_linear, test_r2_linear - train_r2_linear],
    'Median AE': [train_median_ae_linear, test_median_ae_linear, test_median_ae_linear - train_median_ae_linear],
    'MAPE': [train_mape_linear, test_mape_linear, test_mape_linear - train_mape_linear]
}, index=['Train', 'Test', 'Diferencia'])

print("Métricas para el modelo de regresión lineal:")
print(df_metrics_linear)

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
    
    # Calcular R2
    train_r2_laso.append(r2_score(y_train, y_train_pred_laso))
    test_r2_laso.append(r2_score(y_test, y_test_pred_laso))

# Graficar
plt.figure(figsize=(10,6))
plt.plot(alphas, train_r2_laso, label="Entrenamiento")
plt.plot(alphas, test_r2_laso, label="Prueba")
plt.xlabel("Alpha")
plt.ylabel("R2")
plt.legend()
plt.show()

print(train_r2_laso.append(r2_score(y_train, y_train_pred_laso)))
print(test_r2_laso.append(r2_score(y_test, y_test_pred_laso)))

alphas = [0. , 0.2, 0.4, 0.6, 0.8, 1. , 1.2, 1.4, 1.6, 1.8, 2. , 2.2, 2.4]

train_r2_laso = []
test_r2_laso = []

for alpha in alphas:
    laso_modelo = Lasso(alpha=alpha)
    laso_modelo.fit(X_train, y_train)
    
    # Predecir
    y_train_pred_laso = laso_modelo.predict(X_train)
    y_test_pred_laso = laso_modelo.predict(X_test)
    
    # Calcular R2
    train_r2_laso.append(r2_score(y_train, y_train_pred_laso))
    test_r2_laso.append(r2_score(y_test, y_test_pred_laso))

# Graficar
plt.figure(figsize=(10,6))
plt.plot(alphas, train_r2_laso, label="Entrenamiento")
plt.plot(alphas, test_r2_laso, label="Prueba")
plt.xlabel("Alpha")
plt.ylabel("R2")
plt.legend()
plt.show()

#Prueba laso optimizada 
alpha_values = {'alpha': [0.1, 0.5, 1, 5, 10, 20]}
lasso = Lasso()
grid_search = GridSearchCV(lasso, param_grid=alpha_values, cv=5, scoring='r2')
grid_search.fit(X_train, y_train)

# Mejor hiperparámetro encontrado
best_alpha = grid_search.best_params_['alpha']
print(f"Mejor valor de alpha: {best_alpha}")

best_lasso_model = Lasso(alpha=0.3, max_iter=5000000)
best_lasso_model.fit(X_train, y_train)

y_train_lasso = best_lasso_model.predict(X_train)
y_test_lasso = best_lasso_model.predict(X_test)

train_r2_lasso = r2_score(y_train, y_train_lasso)
test_r2_lasso = r2_score(y_test, y_test_lasso)

non_zero_coef = np.sum(best_lasso_model.coef_ != 0)
print(f"Número de variables importantes (no eliminadas): {non_zero_coef}")
print(f"Mejor R2 train (Lasso optimizado): {train_r2_lasso}")
print(f"Mejor R2 en test (Lasso optimizado): {test_r2_lasso}")
print(X_train.head())
print(X_test.head())

#Metricas Modelo Lasso
train_median_ae_lasso = median_absolute_error(y_train, y_train_lasso)
test_median_ae_lasso = median_absolute_error(y_test, y_test_lasso)

train_mape_lasso = mean_absolute_percentage_error(y_train, y_train_lasso) * 100
test_mape_lasso = mean_absolute_percentage_error(y_test, y_test_lasso) * 100

df_metrics_lasso = pd.DataFrame({
    'R2': [train_r2_lasso, test_r2_lasso, test_r2_lasso - train_r2_lasso],
    'Median AE': [train_median_ae_lasso, test_median_ae_lasso, test_median_ae_lasso - train_median_ae_lasso],
    'MAPE': [train_mape_lasso, test_mape_lasso, test_mape_lasso - train_mape_lasso]
}, index=['Train set', 'Test set', 'Diferencia'])

print("Métricas para el modelo Lasso:")
print(df_metrics_lasso)

#Comparación de los dos modelos:
print("Métricas para el modelo de regresión lineal:")
print(df_metrics_linear)

print("Métricas para el modelo Lasso:")
print(df_metrics_lasso)
