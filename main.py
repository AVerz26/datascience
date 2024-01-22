import streamlit as st
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt


def clustering():
    st.title("Técnicas de Análise por Clusterização")

    # Caminho do arquivo Excel pré-determinado
    file_path = "segmentation data.csv"

    # Leitura da planilha
    df = pd.read_csv(file_path)

    # Verificação do conteúdo da planilha
    st.subheader("Tabela de dados")
    st.write("Esta tabela é composta por variáveis quantitativas e qualitativas, e estas últimas precisam ser dummizadas")
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

    # Chame a função para gerar o pewlot
    
    cluster_col = df['cluster']
    centroids = kmeans.cluster_centers_

    # Reverter a escala original dos dados
    X_original_scale = scaler.inverse_transform(X_scaled)

    # Reverter a escala original dos centroides
    centroids_original_scale = scaler.inverse_transform(centroids)

    # Criar o plot usando matplotlib
    plt.scatter(X_original_scale[:, 0], X_original_scale[:, 1], c=cluster_col, cmap='viridis', label='Dados')
    plt.scatter(centroids_original_scale[:, 0], centroids_original_scale[:, 1], c='red', marker='X', s=100, label='Centroids')
    plt.title(f'Clusters identificados pelo K-means (k={optimal_k})')
    plt.xlabel('Feature 1 (age)')
    plt.ylabel('Feature 2 (income)')
    plt.legend()
    st.set_option('deprecation.showPyplotGlobalUse', False)
    # Incorporar o plot no Streamlit
    st.pyplot()



def PCA_f():
    st.title("Análise de Componentes Principais (PCA)")

    # Simulação de dados (substitua isso pelos seus próprios dados)
    data = pd.DataFrame({
        'Feature1': [1, 2, 3, 4, 5],
        'Feature2': [5, 4, 3, 2, 1],
        'Feature3': [10, 8, 6, 4, 2]
    })

    # Exibir dados originais
    st.subheader("Dados Originais:")
    st.write(data)
    
    # Normalizar os dados
    scaler = StandardScaler()
    data_scaled = scaler.fit_transform(data)

    # Aplicar PCA
    pca = PCA()
    pca_result = pca.fit_transform(data_scaled)

    # Exibir a variância explicada acumulada
    explained_variance_ratio = pca.explained_variance_ratio_
    explained_variance_cumulative = explained_variance_ratio.cumsum()

    # Gráfico de variância explicada acumulada
    st.line_chart(explained_variance_cumulative)

    # Número de componentes principais
    num_components = st.slider("Número de Componentes Principais:", 1, min(data.shape), value=2)

    # Exibir gráfico de dispersão 2D com os componentes principais escolhidos
    pca_df = pd.DataFrame(data=pca_result[:, :num_components], columns=[f"PC{i}" for i in range(1, num_components + 1)])
    
    # Criar o plot usando matplotlib
    plt.figure(figsize=(8, 6))
    plt.scatter(pca_df[f"PC1"], pca_df[f"PC2"], label='Dados')
    plt.title(f'Gráfico de Dispersão dos Componentes Principais (PC1 vs PC2)')
    plt.xlabel('Componente Principal 1')
    plt.ylabel('Componente Principal 2')
    plt.legend()
    
    # Incorporar o plot no Streamlit
    st.pyplot()

if __name__ == "__main__":
    st.sidebar.title("Análises Técnicas de Dados")

    add_selectbox = st.sidebar.selectbox(
        "Escolha uma técnica:",
        ("Clustering", "PCA")
    )

    if add_selectbox == "Clustering":
        clustering()
    elif add_selectbox == "PCA":
        PCA_f()


if __name__ == "__main__":
    st.sidebar.title("Análises Técnicas de Dados")

    add_selectbox = st.sidebar.selectbox(
        "Escolha uma técnica:",
        ("Clustering", "PCA")
    )

    if add_selectbox == "Clustering":
        clustering()
    elif add_selectbox == "PCA":
        PCA_f()

