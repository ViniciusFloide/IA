# IMPORTANTE: Antes de começar, instale os pacotes contidos em requirements.txt
#
# python -m pip install -r requirements.txt

import pandas as pd
import joblib
from sklearn.neighbors import KNeighborsClassifier

# IMPORTANTE: Instruções para Repositório de Dados
# 1. Criar um documento no Google Sheet
# 2. Insira dados com estes cabeçalhos: faltas, nota_1, nota_2, resultado
# 3. Adicione de 10 a 15 linhas conforme o exemplo abaixo:
#    8,2.5,4.5,reprovado
#    2,5.0,7.0,aprovado

URL_DADOS = "https://docs.google.com/spreadsheets/d/e/2PACX-1vTjk0oZYvTj98BCepT-It1X14Je-Em3iYdVk1kTB0AS8eN7fx2ov7SHO5UxlVrd4Il4KoSJ65DZdeLt/pub?output=csv"

ARQUIVO_MODELO = 'modelo_desempenho.pkl'


def treinar_modelo():
    print(f"Baixando dados de {URL_DADOS}...")

    data = pd.read_csv(URL_DADOS)

    # Limpa os nomes das colunas
    data.columns = (
        data.columns.astype(str)
        .str.replace('\ufeff', '', regex=False)
        .str.strip()
        .str.lower()
    )

    print("-- Dados Carregados ---")
    print(data.head())

    print("Colunas encontradas:", repr(data.columns.tolist()))

    print("--- Preparando dados para o treino ---")

    features = ["faltas", "nota_1", "nota_2"]
    target = "resultado"

    x = data[features]
    y = data[target]

    # Instancia "modelo" com 3 vizinhos (n_neighbors=3)
    modelo = KNeighborsClassifier(n_neighbors=3)

    modelo.fit(x, y)

    print(f"--- Modelo Treinado! Classes: {modelo.classes_} ---")

    joblib.dump(modelo, ARQUIVO_MODELO)

    print(f"--- Modelo salvo com sucesso em {ARQUIVO_MODELO} ---")


if __name__ == "__main__":
    treinar_modelo()
