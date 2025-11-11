import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import matplotlib.pyplot as plt
import joblib

data = pd.read_csv("ventas_limpio.csv")
print(data.head())

X = data[['Cantidad_Vendida', 'Precio_Unitario', 'Costo_Unitario']]
y = data['Ingreso_Total']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

modelo = LinearRegression()
modelo.fit(X_train, y_train)

print("Coeficientes:", modelo.coef_)
print("Intercepción:", modelo.intercept_)

y_pred = modelo.predict(X_test)

mae = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2 = r2_score(y_test, y_pred)

print("MAE:", mae)
print("RMSE:", rmse)
print("R²:", r2)

plt.figure(figsize=(8,6))
plt.scatter(y_test, y_pred, color='blue', alpha=0.6)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], color='red', linestyle='--')
plt.xlabel('Valores Reales')
plt.ylabel('Valores Predichos')
plt.title('Comparación de Ingreso Total Real vs Predicho')
plt.grid(True)
plt.show()

joblib.dump(modelo, 'modelo_ventas_regresion.pkl')

predicciones = pd.DataFrame({'Ingreso_Real': y_test.values, 'Ingreso_Predicho': y_pred}).head(10)
print(predicciones)
