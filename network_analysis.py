import networkx as nx
import pandas as pd
import numpy as np
import plotly.graph_objects as go

def build_crime_network(df, district=None, limit=80):
    """
    Builds a NetworkX graph connecting Suspects, Victims, Locations, and Crimes.
    Filters by district if specified, and limits nodes for visualization readability.
    """
    # Filter by district if selected
    if district:
        sub_df = df[df["District"] == district]
    else:
        sub_df = df.copy()
        
    # Priority: include repeat offenders and high-risk crimes first to make graph interesting
    sub_df = sub_df.sort_values(by=["Previous_Offenses", "Risk_Level"], ascending=[False, False])
    sub_df = sub_df.head(limit)
    
    G = nx.Graph()
    
    # Store node attributes
    node_types = {}
    node_details = {}
    
    # Add nodes and edges
    for idx, row in sub_df.iterrows():
        fir_node = row["FIR_ID"]
        sus_node = row["Suspect_ID"]
        # Generate a distinct victim node ID based on FIR
        vic_node = f"VIC-{row['FIR_ID'][4:]}"
        loc_node = row["Police_Station"]
        
        # 1. Add Crime incident node
        G.add_node(fir_node)
        node_types[fir_node] = "Crime"
        node_details[fir_node] = f"FIR: {fir_node}<br>Type: {row['Crime_Type']}<br>Loss: Rs. {row['Financial_Loss']:,.2f}<br>Severity: {row['Crime_Severity']}"
        
        # 2. Add Suspect node
        G.add_node(sus_node)
        node_types[sus_node] = "Suspect"
        node_details[sus_node] = f"Suspect: {sus_node}<br>Age: {row['Suspect_Age']}<br>Gender: {row['Suspect_Gender']}<br>Prior Offenses: {row['Previous_Offenses']}"
        
        # 3. Add Victim node
        G.add_node(vic_node)
        node_types[vic_node] = "Victim"
        node_details[vic_node] = f"Victim Profile<br>Age: {row['Victim_Age']}<br>Gender: {row['Victim_Gender']}"
        
        # 4. Add Location node
        G.add_node(loc_node)
        node_types[loc_node] = "Location"
        node_details[loc_node] = f"Location: {loc_node}<br>District: {row['District']}"
        
        # Add edges
        G.add_edge(sus_node, fir_node, relation="Committed")
        G.add_edge(vic_node, fir_node, relation="Victim of")
        G.add_edge(fir_node, loc_node, relation="Occurred at")
        
    # Calculate Centrality metrics
    degree_centrality = nx.degree_centrality(G)
    betweenness_centrality = nx.betweenness_centrality(G)
    
    # Network statistics
    num_nodes = G.number_of_nodes()
    num_edges = G.number_of_edges()
    
    # Identify key suspects (highest centrality)
    suspect_centrality = {node: degree_centrality[node] for node in G.nodes if node_types[node] == "Suspect"}
    top_suspects = sorted(suspect_centrality.items(), key=lambda x: x[1], reverse=True)[:5]
    
    return G, node_types, node_details, degree_centrality, betweenness_centrality, {
        "num_nodes": num_nodes,
        "num_edges": num_edges,
        "top_suspects": top_suspects
    }

def visualize_network_plotly(G, node_types, node_details, degree_centrality):
    """
    Renders the NetworkX graph in an interactive Plotly plot.
    """
    if len(G) == 0:
        fig = go.Figure()
        fig.update_layout(title="No network data available")
        return fig
        
    # Position nodes using spring layout
    pos = nx.spring_layout(G, k=0.15, iterations=50, seed=42)
    
    # Build Edges Traces
    edge_x = []
    edge_y = []
    for edge in G.edges():
        x0, y0 = pos[edge[0]]
        x1, y1 = pos[edge[1]]
        edge_x.extend([x0, x1, None])
        edge_y.extend([y0, y1, None])
        
    edge_trace = go.Scatter(
        x=edge_x, y=edge_y,
        line=dict(width=1, color='#4A5568'),
        hoverinfo='none',
        mode='lines'
    )
    
    # Build Nodes Traces by type
    # Colors: Suspect=Purple (#7209b7), Crime=Cyan (#00b4d8), Victim=Green (#4ABC96), Location=Navy (#1c2541)
    type_colors = {
        "Suspect": "#E63946",   # Red for threat
        "Crime": "#FFB703",     # Orange for incidents
        "Victim": "#52B788",    # Green for victims
        "Location": "#48CAE4"   # Cyan for location nodes
    }
    
    node_traces = []
    
    # Group nodes by type to create separate trace handles for clean legend
    for n_type in ["Suspect", "Crime", "Victim", "Location"]:
        nodes_of_type = [node for node in G.nodes() if node_types.get(node) == n_type]
        if not nodes_of_type:
            continue
            
        node_x = []
        node_y = []
        node_text = []
        node_size = []
        
        for node in nodes_of_type:
            x, y = pos[node]
            node_x.append(x)
            node_y.append(y)
            node_text.append(node_details.get(node, node))
            
            # Size node based on centrality (larger = more connections)
            base_size = 12
            if n_type == "Suspect":
                # Scale up repeat offenders
                size_scaler = degree_centrality.get(node, 0) * 80
                node_size.append(base_size + size_scaler)
            elif n_type == "Location":
                size_scaler = degree_centrality.get(node, 0) * 50
                node_size.append(base_size + size_scaler)
            else:
                node_size.append(base_size)
                
        trace = go.Scatter(
            x=node_x, y=node_y,
            mode='markers',
            name=n_type,
            marker=dict(
                showscale=False,
                color=type_colors[n_type],
                size=node_size,
                line=dict(width=1.5, color='#ffffff' if n_type == 'Suspect' else '#111827')
            ),
            text=node_text,
            hoverinfo='text'
        )
        node_traces.append(trace)
        
    # Combine traces
    fig = go.Figure(
        data=[edge_trace] + node_traces,
        layout=go.Layout(
            showlegend=True,
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1,
                font=dict(color="#ffffff")
            ),
            hovermode='closest',
            margin=dict(b=10, l=10, r=10, t=40),
            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)'
        )
    )
    
    return fig
