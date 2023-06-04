import pandas as pd

# Criar DataFrame
data = {0: [1, 2, 3], 1: [5, 6, 7]}
df = pd.DataFrame(data)

# Salvar DataFrame como arquivo CSV
df.to_csv('meu_dataframe.csv', index=False)

# Ler o arquivo CSV como um novo DataFrame
df_lido = pd.read_csv('meu_dataframe.csv', dtype={0: int, 1: int})

new_df = pd.DataFrame()
array_teste = df_lido.iloc[0].values
# for i in range(len(df_lido)):
#     data = df_lido.iloc[i].values
#     new_df[i] = data
# Iterar sobre as linhas do DataFrame original
i = 0
for _, linha in df_lido.iterrows():
    # Adicionar a linha ao novo DataFrame
    array = df_lido.iloc[i].to_numpy()
    new_df[i] = array.reshape(-1).transpose()
    i += 1
    # new_df = pd.concat([new_df, ], axis=1)

# Converter as colunas do DataFrame para listas
colunas_df = df.values.flatten().tolist()
colunas_df_lido = df_lido.values.flatten().tolist()
new_df = pd.DataFrame(df_lido.values)

if new_df.equals(df):
    print("igual")
else:
    print("não igual")

# Exibir DataFrame lido
print(df_lido)
