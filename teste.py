import matplotlib.pyplot as plt
import pandas as pd

# Criar DataFrame de exemplo
data = {'Grupo 1': [1, 2, 3, 4, 5],
        'Grupo 2': [2, 4, 6, 8, 10],
        'Grupo 3': [3, 6, 9, 12, 15]}
df = pd.DataFrame(data)

# Plotar boxplot usando matplotlib
plt.boxplot(df.values)

# Configurar rótulos dos eixos
plt.xticks(range(1, len(df.columns) + 1), df.columns)
plt.ylabel('Valores')

# Exibir o gráfico
plt.show()
