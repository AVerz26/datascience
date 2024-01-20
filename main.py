import streamlit as st
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

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

    # Chame a função para gerar o plot
    visualize_clusters(X_scaled, df['cluster'], optimal_k)

def visualize_clusters(X_scaled, cluster_col, optimal_k):
    # Criar o plot usando matplotlib
    plt.scatter(X_scaled[:, 0], X_scaled[:, 1], c=cluster_col, cmap='viridis')
    plt.title(f'Clusters identificados pelo K-means (k={optimal_k})')
    plt.xlabel('Feature 1 (age)')
    plt.ylabel('Feature 2 (income)')
    st.set_option('deprecation.showPyplotGlobalUse', False)
    # Incorporar o plot no Streamlit
    st.pyplot()

if __name__ == "__main__":
    main()
