import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Supondo que você tenha seus dados em um formato similar ao apresentado
data = {
    'rank': [1, 2, 3, 4, 5],
    'channel_info': ['cristiano', 'kyliejenner', 'leomessi', 'selenagomez', 'therock'],
    'influence_score': [92, 91, 90, 93, 91],
    'posts': ['3.3k', '6.9k', '0.89k', '1.8k', '6.8k'],
    'followers': [475800000.0, 366200000.0, 357300000.0, 342700000.0, 334100000.0],
    'avg_likes': ['8700000.0', '8300000.0', '6800000.0', '6200000.0', '1900000.0'],
    '60_day_eng_rate': ['0.0139', '0.0162', '0.0124', '0.0097', '0.0020'],
    'new_post_avg_like': ['6.5m', '5.9m', '4.4m', '3.3m', '665.3k'],
    'total_likes': ['2.900000e+10', '5.740000e+10', '6.000000e+09', '1.150000e+10', '1.250000e+10'],
    'country': ['PT', 'US', 'AR', 'US', 'US']
}

df = pd.DataFrame(data)

# Função para converter strings para float
def convert_to_float(value):
    if isinstance(value, str):
        if 'k' in value:
            return float(value.replace('k', '').replace('.', '').replace(',', '.')) * 1e3
        elif 'm' in value:
            return float(value.replace('m', '').replace('.', '').replace(',', '.')) * 1e6
        else:
            return float(value)
    return value

# Aplicando a função de conversão nas colunas necessárias
columns_to_convert = ['posts', 'avg_likes', '60_day_eng_rate', 'new_post_avg_like', 'total_likes']
for column in columns_to_convert:
    df[column] = df[column].apply(convert_to_float)

# Transformação da coluna country
continent_map = {
    'AR': 1,  # América do Sul
    'US': 20, # América do Norte
    'PT': 40  # Europa
}
df['continent'] = df['country'].map(continent_map)

# Análise Exploratória
plt.figure(figsize=(10, 5))
sns.scatterplot(x='followers', y='avg_likes', hue='60_day_eng_rate', size='influence_score', sizes=(20, 200), data=df)
plt.title('Relação entre Followers e Avg Likes')
plt.xlabel('Followers')
plt.ylabel('Avg Likes')
plt.tight_layout()
plt.show()

plt.figure(figsize=(10, 5))
sns.barplot(x='rank', y='influence_score', data=df)
plt.title('Rank vs Influence Score')
plt.xlabel('Rank')
plt.ylabel('Influence Score')
plt.tight_layout()
plt.show()

# Preparação para o kNN
X = df[['followers', 'avg_likes', '60_day_eng_rate', 'total_likes', 'continent']]
y = df['influence_score']

# Divisão dos dados
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Normalização dos dados
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Implementação do kNN
knn = KNeighborsRegressor()
param_grid = {'n_neighbors': range(1, 11)}  # Testando valores de k de 1 a 10
grid_search = GridSearchCV(knn, param_grid, cv=2)  # Reduzindo o número de splits
grid_search.fit(X_train_scaled, y_train)

# Melhor valor de k
best_k = grid_search.best_params_['n_neighbors']
print(f'Melhor valor de k: {best_k}')

# Avaliação do modelo
y_pred = grid_search.predict(X_test_scaled)

# Cálculo das métricas de avaliação
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)

print(f'MAE: {mae}')
print(f'MSE: {mse}')
print(f'RMSE: {rmse}')
print(f'R²: {r2}')
