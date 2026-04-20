import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

# Ma'lumotlar manbasidan ma'lumotlar olinadi
data = {
    'Sana': ['2022-01-01', '2022-01-02', '2022-01-03', '2022-01-04', '2022-01-05'],
    'Sotuv': [100, 120, 110, 130, 140],
    'Kirim': [50, 60, 70, 80, 90]
}

# Pandas kutubxonasidan ma'lumotlar olinadi
df = pd.DataFrame(data)

# Ma'lumotlarni o'rganish
print(df.head())

# Ma'lumotlarni tayyorlash
df['Sana'] = pd.to_datetime(df['Sana'])
df['Sotuv'] = df['Sotuv'].astype(float)
df['Kirim'] = df['Kirim'].astype(float)

# Grafik chizish
plt.figure(figsize=(10, 6))
plt.plot(df['Sana'], df['Sotuv'], label='Sotuv')
plt.plot(df['Sana'], df['Kirim'], label='Kirim')
plt.xlabel('Sana')
plt.ylabel('Ma'lumot')
plt.title('Sotuv va Kirim Grafiki')
plt.legend()
plt.show()

# Modelni o'rganish
X = df[['Sotuv']]
y = df['Kirim']

# Modelni tayyorlash
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = LinearRegression()
model.fit(X_train, y_train)

# Modelni baholash
y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
print(f'Modelning bahosi: {mse}')

# Modelni ishlatish
sotuv = 150
kirimi = model.predict([[sotuv]])
print(f'150 sotuvdan {kirimi[0]} kirim olinadi')
