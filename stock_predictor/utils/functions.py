import pandas as pd
import numpy as np
import torch
from torch_geometric_temporal.signal import DynamicGraphTemporalSignal
import networkx as nx
import matplotlib.pyplot as plt

def build_dynamic_graph(df, correlation_threshold=0.5):
    """
    Convierte un DataFrame donde cada columna es una acción y cada fila es un instante de tiempo
    en un objeto DynamicGraphTemporalSignal para predecir el precio de cierre futuro.
    
    Parámetros:
    - df: DataFrame con índice temporal y columnas por acción.
    - correlation_threshold: Mínima correlación para crear aristas entre nodos.
    
    Retorna:
    - objeto DynamicGraphTemporalSignal listo para usar.
    """
    # Preparar los datos
    df = df.sort_index()  # Asegurarse de que esté ordenado por tiempo
    tickers = df.columns  # Empresas como columnas
    time_steps = df.index  # Índice temporal (fechas)

    features, edge_indices, edge_weights, targets = [], [], [], []

    # Construir el grafo para cada t
    for t in range(len(time_steps) - 1):  # Excluyendo el último tiempo para los targets
        # Filtrar datos para el tiempo actual y el siguiente
        current_time = time_steps[t]
        next_time = time_steps[t + 1]
        current_df = df.loc[current_time].values  # Precios de cierre para ese tiempo
        next_df = df.loc[next_time].values  # Precios de cierre para el siguiente tiempo
        
        # Nodos: Precios de cierre actuales como características
        features.append(torch.tensor(current_df, dtype=torch.float).view(-1, 1))  # (n_nodos, n_features)
        
        # Targets: Precios de cierre en el siguiente tiempo
        targets.append(torch.tensor(next_df, dtype=torch.float).view(-1, 1))

        # Construir aristas entre empresas por correlación en ese tiempo
        corr_matrix = np.corrcoef(current_df)  # Matriz de correlación entre las empresas
        edge_index_spatial = []
        edge_weight_spatial = []
        
        for i in range(len(tickers)):
            for j in range(i + 1, len(tickers)):
                corr_value = corr_matrix[i, j]
                if corr_value > correlation_threshold:
                    edge_index_spatial.append([i, j])
                    edge_index_spatial.append([j, i])  # Grafo no dirigido
                    edge_weight_spatial.append(corr_value)
                    edge_weight_spatial.append(corr_value)
        
        # Conexiones temporales (nodo consigo mismo en el siguiente tiempo)
        edge_index_temporal = [[i, i] for i in range(len(tickers))]
        edge_weight_temporal = [0.0] * len(tickers)  # Peso 0 para la conexión temporal
        
        # Combinar las aristas espaciales y temporales
        edge_index_t = edge_index_spatial + edge_index_temporal
        edge_weight_t = edge_weight_spatial + edge_weight_temporal
        
        edge_indices.append(torch.tensor(edge_index_t, dtype=torch.long).t().contiguous())
        edge_weights.append(torch.tensor(edge_weight_t, dtype=torch.float))
    
    return DynamicGraphTemporalSignal(
        edge_indices=edge_indices,
        edge_weights=edge_weights,
        features=features,
        targets=targets
    )

def visualize_dynamic_graph(temporal_signal, time_step=0):
    """
    Visualiza el grafo temporal en un instante de tiempo específico.
    
    Parámetros:
    - temporal_signal: El objeto DynamicGraphTemporalSignal.
    - time_step: El índice del tiempo para visualizar (por ejemplo, 0 para el primer tiempo).
    """
    snapshot = temporal_signal[time_step]
    edge_index = snapshot.edge_index  # Las aristas del grafo en el tiempo step
    edge_weights = snapshot.edge_attr  # Pesos de las aristas
    features = snapshot.x  # Características de los nodos (precios de cierre)

    # Crear el grafo utilizando NetworkX
    G = nx.Graph()

    # Agregar nodos
    for i in range(features.shape[0]):
        G.add_node(i, label=f"Empresa {i}", feature=features[i].item())

    # Agregar aristas con pesos
    for i in range(edge_index.shape[1]):
        source, target = edge_index[0][i].item(), edge_index[1][i].item()
        weight = edge_weights[i].item()
        G.add_edge(source, target, weight=weight)

    # Dibujar el grafo
    pos = nx.spring_layout(G)  # Layout para la posición de los nodos
    plt.figure(figsize=(10, 8))
    
    # Dibujar nodos y aristas
    nx.draw_networkx_nodes(G, pos, node_size=500, node_color='skyblue', alpha=0.7)
    nx.draw_networkx_edges(G, pos, width=2, alpha=0.5, edge_color='grey')

    # Etiquetas de nodos (empresas)
    nx.draw_networkx_labels(G, pos, font_size=12, font_weight='bold', font_color='black')

    # Etiquetas de aristas (pesos)
    edge_labels = nx.get_edge_attributes(G, 'weight')
    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_size=10)

    # Mostrar el grafo
    plt.title(f"Grafo Temporal en t={time_step}")
    plt.axis('off')
    plt.show()

