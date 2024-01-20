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
    selected_columns = ['Age', 'Income', 'Education', 'Occupation', 'Settlement size']
    

    # Análise de clusterização
    st.subheader("Análise de Clusterização")

    # Criando um DataFrame apenas com as colunas selecionadas
    X = df[selected_columns]

    # Dummizando as colunas 'education', 'occupation' e 'settlement_size'
    X = pd.get_dummies(X, columns=['Education', 'Occupation', 'Settlement size'], drop_first=True)

    # Normalizando os dados
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    
    # Aplicação do algoritmo KMeans
    n_clusters = st.slider("Número de Clusters (k)", min_value=2, max_value=10, value=3)
    optimal_k = n_clusters  # Ajuste conforme a visualização da curva do cotovelo

    # Aplicando o K-means com o número ótimo de clusters
    kmeans = KMeans(n_clusters=optimal_k, random_state=42)
    df['cluster'] = kmeans.fit_predict(X_scaled)

    # Visualização dos clusters
    st.subheader("Visualização dos Clusters")
    visualize_clusters(X_scaled, X_scaled[:,1], 'cluster')

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
