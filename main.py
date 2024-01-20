import streamlit as st
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns
import sklearn

def main():
    st.title("Análise de Clusterização com KMeans")

    # Upload da planilha
    uploaded_file = st.file_uploader("Escolha um arquivo em Excel (xlsx)", type="xlsx")

    if uploaded_file is not None:
        # Leitura da planilha
        df = pd.read_excel(uploaded_file)

        # Verificação do conteúdo da planilha
        st.subheader("Conteúdo da Planilha:")
        st.write(df)

        # Separação de valores categóricos e quantitativos
        categorical_columns = df.select_dtypes(include='object').columns
        numerical_columns = df.select_dtypes(exclude='object').columns

        st.subheader("Colunas Categóricas:")
        st.write(categorical_columns)

        st.subheader("Colunas Quantitativas:")
        selected_numerical_columns = st.multiselect("Selecione as colunas quantitativas", numerical_columns, default=numerical_columns)

        # Análise de clusterização
        st.subheader("Análise de Clusterização")

        # Parâmetros ajustáveis
        n_clusters = st.slider("Número de Clusters (k)", min_value=2, max_value=10, value=3)

        # Tratamento dos dados (preenchimento de valores ausentes e normalização)
        df[selected_numerical_columns] = df[selected_numerical_columns].fillna(0)
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(df[selected_numerical_columns])

        # Aplicação do algoritmo KMeans
        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        df['cluster'] = kmeans.fit_predict(X_scaled)

        # Visualização dos clusters
        st.subheader("Visualização dos Clusters")
        visualize_clusters(df, selected_numerical_columns, 'cluster')

def visualize_clusters(df, features, cluster_col):
    # Pairplot para visualização dos clusters
    sns.pairplot(df, hue=cluster_col, palette='viridis', diag_kind='kde')
    st.pyplot()

    # Visualização da média das variáveis por cluster
    st.subheader("Média das Variáveis por Cluster")
    cluster_means = df.groupby(cluster_col)[features].mean()
    st.write(cluster_means)

if __name__ == "__main__":
    main()
