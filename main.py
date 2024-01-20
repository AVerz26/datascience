import streamlit as st
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns

def main():
    st.title("Técnicas de Análise por Clusterização")

    # Caminho do arquivo Excel pré-determinado
    file_path = "segmentation data.csv"

    # Leitura da planilha
    df = pd.read_csv(file_path)

    # Verificação do conteúdo da planilha
    st.subheader("Conteúdo da Planilha:")
    st.write(df)

    # Seleção de colunas categóricas e quantitativas
    categorical_columns = ['Sex', 'Marital status', 'Education', 'Occupation', 'Settlement size']
    numerical_columns = ['Age', 'Income']

    st.subheader("Colunas Categóricas:")
    st.write(categorical_columns)

    st.subheader("Colunas Quantitativas:")
    selected_numerical_columns = numerical_columns

    # Análise de clusterização
    st.subheader("Análise de Clusterização")

    # Tratamento dos dados (preenchimento de valores ausentes e normalização)
    df[selected_numerical_columns] = df[selected_numerical_columns].fillna(0)

    # Dummizar variáveis categóricas
    df_dummies = pd.get_dummies(df['Education', 'Occupation', 'Settlement size'], drop_first=True)
    df = pd.concat([df, df_dummies], axis=1)

    # Selecionar as colunas corretas após a dummização
    selected_columns_after_dummization = selected_numerical_columns + list(df_dummies.columns)

    # Escalonamento das variáveis
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(df[selected_columns_after_dummization])

    # Aplicação do algoritmo KMeans
    n_clusters = st.slider("Número de Clusters (k)", min_value=2, max_value=10, value=3)
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    df['cluster'] = kmeans.fit_predict(X_scaled)

    # Visualização dos clusters
    st.subheader("Visualização dos Clusters")
    visualize_clusters(df, selected_columns_after_dummization, 'cluster')

def visualize_clusters(df, features, cluster_col):
    # Pairplot para visualização dos clusters
    pairplot_data = pd.concat([df[features], df[cluster_col]], axis=1)
    sns.pairplot(pairplot_data, hue=cluster_col, palette='viridis', diag_kind='kde')
    st.pyplot()

    # Visualização da média das variáveis por cluster
    st.subheader("Média das Variáveis por Cluster")
    cluster_means = df.groupby(cluster_col)[features].mean()
    st.write(cluster_means)


if __name__ == "__main__":
    main()
