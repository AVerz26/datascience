import streamlit as st
import pychrono as chrono
import numpy as np
import time

# Função para criar o sistema de simulação
def criar_sistema():
    system = chrono.ChSystem()  # Sistema de dinâmica não suave
    
    # Criando um corpo fixo (terra)
    ground = chrono.ChBodyAuxRef()
    system.Add(ground)
    
    # Criando o corpo rígido (uma caixa)
    box = chrono.ChBodyAuxRef()
    box.SetMass(1.0)
    box.SetInertiaXX(chrono.ChVector(1, 1, 1))
    box.SetPos(chrono.ChVector(0, 1, 0))  # posição inicial
    box.SetBodyFixed(False)
    system.Add(box)
    
    return system, box

# Função para simulação e visualização
def simular():
    system, box = criar_sistema()
    
    # Configuração do Streamlit
    st.title("Simulação de Corpo Rígido com PyChrono e Streamlit")
    
    # Parâmetros de tempo
    tempo_simulacao = st.slider("Tempo de Simulação (segundos)", 1, 10, 5)
    passos_simulacao = st.slider("Passos por Segundo", 10, 100, 50)
    
    # Simulação do movimento
    st.write("Iniciando simulação...")
    tempo_inicial = time.time()
    passos_por_segundo = passos_simulacao
    tempo_total = tempo_simulacao
    dt = 1.0 / passos_por_segundo  # Intervalo de tempo
    
    # Loop de simulação
    posicoes = []  # Armazenar posições do corpo ao longo do tempo
    for i in range(int(tempo_total * passos_por_segundo)):
        system.DoStepDynamics(dt)
        pos = box.GetPos()
        posicoes.append((pos.x, pos.y, pos.z))
    
    # Exibir resultados
    st.write(f"Posições ao longo do tempo (último tempo: {tempo_total} segundos):")
    st.line_chart(np.array(posicoes))
    
    # Exibindo a posição final
    st.write(f"Posição final da caixa: x={pos.x:.2f}, y={pos.y:.2f}, z={pos.z:.2f}")

# Chamar a função de simulação
if __name__ == "__main__":
    simular()
