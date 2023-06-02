import pandas as pd

# Criar DataFrame
data = {0: [1, 2, 3], 1: [5, 6, 7]}
df = pd.DataFrame(data)

# Salvar DataFrame como arquivo CSV
df.to_csv('meu_dataframe.csv', index=False)

# Ler o arquivo CSV como um novo DataFrame
df_lido = pd.read_csv('meu_dataframe.csv', dtype={0: int, 1: int})

# Converter as colunas do DataFrame para listas
colunas_df = df.values.flatten().tolist()
colunas_df_lido = df_lido.values.flatten().tolist()
new_df = pd.DataFrame(df_lido)

if new_df.equals(df):
    print("igual")
else:
    print("não igual")

# Exibir DataFrame lido
print(df_lido)
