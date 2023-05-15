import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_squared_error

# Создание датасета
np.random.seed(42)
data = pd.DataFrame({
    'Sales': np.random.normal(10, 1, 400),
    'Competitor_Price': np.random.normal(10, 2, 400),
    'Income': np.random.normal(60, 10, 400),
    'Advertising': np.random.normal(5, 1, 400),
    'Population': np.random.normal(200, 20, 400),
    'Price': np.random.normal(10, 2, 400),
    'Shelf_Location': np.random.choice(['Bad', 'Medium', 'Good'], 400),
    'Age': np.random.normal(45, 10, 400),
    'Education': np.random.normal(15, 2, 400),
    'Urban': np.random.choice(['Yes', 'No'], 400),
    'US': np.random.choice(['Yes', 'No'], 400),
})

# Замена категориальных переменных с помощью one-hot encoding
data = pd.get_dummies(data, columns=['Shelf_Location', 'Urban', 'US'])

# Разделение данных на признаки (X) и целевую переменную (y)
X = data.drop('Sales', axis=1)
y = data['Sales']

# Разделение данных на обучающую и тестовую выборки
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Создание модели случайного леса
rf = RandomForestRegressor(n_estimators=100, random_state=42)

# Обучение модели
rf.fit(X_train, y_train)

# Предсказания
y_pred = rf.predict(X_test)

# Вычисление среднеквадратичной ошибки
rmse_rf = np.sqrt(mean_squared_error(y_test, y_pred))
print("Random Forest RMSE:", rmse_rf)

# Создание модели градиентного бустинга
gb = GradientBoostingRegressor(n_estimators=100, learning_rate=0.1, random_state=42)

# Обучение модели градиентного бустинга
gb.fit(X_train, y_train)

# Предсказания для модели градиентного бустинга
y_pred = gb.predict(X_test)

# Вычисление среднеквадратичной ошибки для градиентного бустинга
rmse_gb = np.sqrt(mean_squared_error(y_test, y_pred))
print("Gradient Boosting RMSE:", rmse_gb)

# Сохранение результатов в Excel
results = {"Model": ["Random Forest", "Gradient Boosting"], "RMSE": [rmse_rf, rmse_gb]}
results_df = pd.DataFrame(results)
results_df.to_excel("results.xlsx")
