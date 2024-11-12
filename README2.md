# Relatório Técnico: Implementação e Análise do Algoritmo k-Nearest Neighbors (kNN) Aplicado ao Instagram

**Nome do Residente:** [Seu Nome]  
**Data de Entrega:** [Data de Entrega]

## Resumo

Este projeto tem como objetivo implementar e analisar o algoritmo k-Nearest Neighbors (kNN) aplicado a um conjunto de dados de influenciadores do Instagram. A análise se concentra na previsão do "influence_score" com base em variáveis como seguidores, engajamento e métricas de postagem. Utilizamos a biblioteca Scikit-Learn para implementar o modelo, otimizando seus parâmetros e avaliando seu desempenho por meio de métricas como MAE, MSE e R².

## Introdução

O uso de influenciadores nas redes sociais tem crescido significativamente, e entender seu impacto e influência pode ser crucial para marcas e profissionais de marketing. O kNN é um algoritmo de aprendizado de máquina simples, mas eficaz, que pode ser usado para prever valores contínuos, como o "influence_score", com base em dados históricos.

### Conjunto de Dados

O conjunto de dados utilizado neste projeto contém as seguintes variáveis:
- **rank:** Rank do influenciador.
- **channel_info:** Nome do influenciador.
- **influence_score:** Pontuação de influência do influenciador.
- **posts:** Número de postagens.
- **followers:** Número de seguidores.
- **avg_likes:** Média de curtidas por postagem.
- **60_day_eng_rate:** Taxa de engajamento nos últimos 60 dias.
- **new_post_avg_like:** Média de curtidas das novas postagens.
- **total_likes:** Total de curtidas acumuladas.
- **country:** País de origem do influenciador.

Além disso, a coluna **country** foi transformada em um valor numérico representando continentes.

## Metodologia

### Análise Exploratória

A análise inicial dos dados revelou informações importantes sobre a distribuição de seguidores e curtidas, além de possíveis correlações entre variáveis. Utilizamos gráficos de dispersão para observar a relação entre seguidores e média de curtidas, assim como gráficos de barras para comparar o rank com a pontuação de influência.

### Implementação do Algoritmo

O algoritmo kNN foi implementado utilizando a biblioteca Scikit-Learn. Os dados foram normalizados e o modelo foi ajustado com validação cruzada. A transformação da variável **country** foi realizada para categorizá-la em continentes, facilitando a análise.

### Validação e Ajuste de Hiperparâmetros

Utilizamos o GridSearchCV para otimizar os hiperparâmetros do modelo, testando diferentes valores de k. A validação cruzada foi utilizada para garantir a consistência e a performance do modelo.

## Resultados

### Métricas de Avaliação

Após a implementação do modelo kNN, as seguintes métricas foram obtidas:

- **MAE:** [Valor MAE]
- **MSE:** [Valor MSE]
- **RMSE:** [Valor RMSE]
- **R²:** [Valor R²]

### Visualizações

- Gráfico de dispersão de seguidores vs. média de curtidas.
- Gráfico de barras comparando rank com pontuação de influência.

```python
# Exemplo de gráfico
import matplotlib.pyplot as plt
import seaborn as sns

plt.figure(figsize=(10, 5))
sns.scatterplot(x='followers', y='avg_likes', hue='60_day_eng_rate', data=df)
plt.title('Relação entre Followers e Avg Likes')
plt.xlabel('Followers')
plt.ylabel('Avg Likes')
plt.show()
