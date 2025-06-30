import json, pandas as pd, streamlit as st
import altair as alt                     # chart
import plotly.graph_objects as go        # network graph
import networkx as nx                    # network analysis
import numpy as np
from collections import Counter

# ---------- Load snapshot ----------
with open("cluster_data.json") as f:
    raw = json.load(f)
df = pd.json_normalize(raw)              # 1 row per cluster

st.title("🔬 Cluster Trend Explorer")
st.markdown("### Snapshot Explorer - Research Cluster Analysis")

# ---------- Sidebar controls ----------
st.sidebar.header("📊 Cluster Selection")

# Sort clusters by size (descending) for the dropdown
df_sorted = df.sort_values('size', ascending=False)
cluster_options = [f"Cluster {row['cluster_id']} (size: {row['size']}) - {row['label'][:50]}..." 
                  if len(row['label']) > 50 else f"Cluster {row['cluster_id']} (size: {row['size']}) - {row['label']}"
                  for _, row in df_sorted.iterrows()]
cluster_ids_sorted = df_sorted['cluster_id'].tolist()

selected_idx = st.sidebar.selectbox(
    "Choose a cluster (sorted by size):", 
    range(len(cluster_options)),
    format_func=lambda x: cluster_options[x]
)
selected = cluster_ids_sorted[selected_idx]

# ---------- Key metrics ----------
row = df[df.cluster_id == selected].iloc[0]
st.header(f"Cluster {selected}")
st.subheader(f"📝 {row['label']}")

# Metrics in a nice layout
col1, col2, col3 = st.columns(3)
with col1:
    st.metric("📊 Cluster Size", int(row["size"]), help="Number of papers in this cluster")
with col2:
    avg_pagerank = round(sum(row["pageranks"])/len(row["pageranks"]), 6) if row["pageranks"] else 0
    st.metric("🔗 Avg PageRank", f"{avg_pagerank:.6f}", help="Average PageRank score of papers")
with col3:
    avg_betweenness = round(sum(row["betweenness"])/len(row["betweenness"]), 2) if row["betweenness"] else 0
    st.metric("🌐 Avg Betweenness", f"{avg_betweenness:.2f}", help="Average betweenness centrality")

# ---------- Paper list (optional) ----------
if row["papers"] and len(row["papers"]) > 0:
    with st.expander(f"📄 Papers in Cluster {selected} ({len(row['papers'])} papers)"):
        for i, paper_url in enumerate(row["papers"], 1):
            st.write(f"{i}. [{paper_url}]({paper_url})")
else:
    st.info("No papers listed for this cluster.")

# ---------- Authors information ----------
if row["authors"] and len(row["authors"]) > 0:
    with st.expander(f"👥 Authors in Cluster {selected} ({len(row['authors'])} authors)"):
        # Display authors in a more readable format
        authors_per_row = 3
        author_list = row["authors"]
        for i in range(0, len(author_list), authors_per_row):
            cols = st.columns(authors_per_row)
            for j, author in enumerate(author_list[i:i+authors_per_row]):
                with cols[j]:
                    st.write(f"• {author}")

# ---------- Global bar chart of cluster sizes ----------
st.header("📈 Global Cluster Size Distribution")

# Create a better chart with colors and sorting
chart_data = df.copy()
chart_data['selected'] = chart_data['cluster_id'] == selected

chart = (
    alt.Chart(chart_data)
    .mark_bar()
    .encode(
        x=alt.X("cluster_id:O", title="Cluster ID", sort=alt.SortField('size', order='descending')),
        y=alt.Y("size:Q", title="Cluster Size"),
        color=alt.condition(
            alt.datum.selected,
            alt.value('red'),  # Selected cluster in red
            alt.value('steelblue')  # Other clusters in blue
        ),
        tooltip=["cluster_id:O", "label:N", "size:Q"]
    )
    .properties(
        width=800,
        height=400,
        title="Cluster Sizes (Selected cluster highlighted in red)"
    )
)

st.altair_chart(chart, use_container_width=True)

# ---------- Interactive Network Graph ----------
st.header("🕸️ Cluster Network Graph")
st.markdown("Interactive force-directed network showing cluster relationships based on shared authors")

# Network controls (moved outside cached function)
min_shared_authors = st.sidebar.slider("Min shared authors for connection", 1, 10, 1)

@st.cache_data
def create_cluster_network(df, min_shared_authors):
    """Create a network graph of clusters based on shared authors"""
    G = nx.Graph()
    
    # Add nodes (clusters) with attributes
    for _, row in df.iterrows():
        G.add_node(
            row['cluster_id'],
            size=row['size'],
            label=row['label'],
            authors=set(row['authors']) if row['authors'] else set(),
            pagerank_avg=np.mean(row['pageranks']) if row['pageranks'] else 0,
            betweenness_avg=np.mean(row['betweenness']) if row['betweenness'] else 0
        )
    
    # Add edges based on shared authors (minimum threshold for connection)
    nodes = list(G.nodes())
    for i in range(len(nodes)):
        for j in range(i + 1, len(nodes)):
            node1, node2 = nodes[i], nodes[j]
            authors1 = G.nodes[node1]['authors']
            authors2 = G.nodes[node2]['authors']
            
            shared_authors = len(authors1.intersection(authors2))
            if shared_authors >= min_shared_authors:
                G.add_edge(node1, node2, weight=shared_authors)
    
    return G

# Create the network
G = create_cluster_network(df, min_shared_authors)

# Network layout options
layout_option = st.sidebar.selectbox(
    "Network Layout",
    ["spring_spread", "spring", "random", "circular", "shell"],
    index=0
)

if layout_option == "spring":
    pos = nx.spring_layout(G, k=1, iterations=100, seed=42)
elif layout_option == "spring_spread":
    # Better spring layout for spreading nodes
    pos = nx.spring_layout(G, k=3, iterations=200, seed=42)
elif layout_option == "circular":
    pos = nx.circular_layout(G)
elif layout_option == "random":
    pos = nx.random_layout(G, seed=42)
else:  # shell
    pos = nx.shell_layout(G)

# If there are no edges (disconnected graph), use a better layout
if len(G.edges()) == 0:
    # Use random layout for disconnected nodes to spread them out
    pos = nx.random_layout(G, seed=42)
elif len(G.edges()) < len(G.nodes()) * 0.1:  # Very sparse graph
    # Use spring layout with high repulsion to spread nodes
    pos = nx.spring_layout(G, k=3, iterations=300, seed=42)

# Create Plotly network visualization
def create_network_plot(G, pos, selected_cluster):
    # Extract edges
    edge_x = []
    edge_y = []
    edge_info = []
    
    for edge in G.edges():
        x0, y0 = pos[edge[0]]
        x1, y1 = pos[edge[1]]
        edge_x.extend([x0, x1, None])
        edge_y.extend([y0, y1, None])
        
        # Edge info for hover
        weight = G.edges[edge].get('weight', 1)
        edge_info.append(f"Clusters {edge[0]} ↔ {edge[1]}<br>Shared authors: {weight}")
    
    # Create edge trace
    edge_trace = go.Scatter(
        x=edge_x, y=edge_y,
        line=dict(width=0.5, color='#888'),
        hoverinfo='none',
        mode='lines'
    )
    
    # Extract nodes
    node_x = []
    node_y = []
    node_text = []
    node_size = []
    node_color = []
    node_info = []
    
    for node in G.nodes():
        x, y = pos[node]
        node_x.append(x)
        node_y.append(y)
        
        # Node attributes
        size = G.nodes[node]['size']
        label = G.nodes[node]['label']
        pagerank_avg = G.nodes[node]['pagerank_avg']
        betweenness_avg = G.nodes[node]['betweenness_avg']
        
        # Scale node size for visualization (min 10, max 50)
        scaled_size = max(10, min(50, size * 2))
        node_size.append(scaled_size)
        
        # Color based on whether it's selected
        if node == selected_cluster:
            node_color.append('red')
        else:
            node_color.append('lightblue')
        
        # Node text and hover info
        node_text.append(f"C{node}")
        node_info.append(
            f"<b>Cluster {node}</b><br>"
            f"Topic: {label}<br>"
            f"Size: {size} papers<br>"
            f"Avg PageRank: {pagerank_avg:.6f}<br>"
            f"Avg Betweenness: {betweenness_avg:.2f}<br>"
            f"Connections: {len(list(G.neighbors(node)))}"
        )
    
    # Create node trace
    node_trace = go.Scatter(
        x=node_x, y=node_y,
        mode='markers+text',
        hoverinfo='text',
        text=node_text,
        textposition="middle center",
        hovertext=node_info,
        marker=dict(
            size=node_size,
            color=node_color,
            line=dict(width=2, color='black'),
            opacity=0.8
        )
    )
    
    # Create the figure
    fig = go.Figure(data=[edge_trace, node_trace],
                   layout=go.Layout(
                       title=dict(
                           text=f'Cluster Network Graph ({len(G.nodes())} clusters, {len(G.edges())} connections)',
                           font=dict(size=16)
                       ),
                       showlegend=False,
                       hovermode='closest',
                       margin=dict(b=20,l=5,r=5,t=40),
                       annotations=[ dict(
                           text="Node size = cluster size | Red = selected cluster | Hover for details",
                           showarrow=False,
                           xref="paper", yref="paper",
                           x=0.005, y=-0.002,
                           xanchor='left', yanchor='bottom',
                           font=dict(color="gray", size=12)
                       )],
                       xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                       yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                       height=600
                   ))
    
    return fig

# Display network statistics
col1, col2, col3 = st.columns(3)
with col1:
    st.metric("Network Nodes", len(G.nodes()))
with col2:
    st.metric("Network Edges", len(G.edges()))
with col3:
    if len(G.nodes()) > 0:
        avg_degree = sum(dict(G.degree()).values()) / len(G.nodes())
        st.metric("Avg Connections", f"{avg_degree:.1f}")
    else:
        st.metric("Avg Connections", "0")

# Create and display the network plot
if len(G.nodes()) > 0:
    network_fig = create_network_plot(G, pos, selected)
    st.plotly_chart(network_fig, use_container_width=True)
    
    # Network insights
    if selected in G.nodes():
        neighbors = list(G.neighbors(selected))
        if neighbors:
            st.info(f"🔗 Cluster {selected} is connected to {len(neighbors)} other clusters: {', '.join(map(str, neighbors))}")
        else:
            st.info(f"🔗 Cluster {selected} has no connections in the current network (try lowering the minimum shared authors threshold)")
    
else:
    st.warning("No network connections found with current settings. Try lowering the minimum shared authors threshold.")

# ---------- Summary statistics ----------
st.header("📋 Dataset Summary")
col1, col2, col3, col4 = st.columns(4)
with col1:
    st.metric("Total Clusters", len(df))
with col2:
    st.metric("Total Papers", df['size'].sum())
with col3:
    st.metric("Largest Cluster", df['size'].max())
with col4:
    st.metric("Avg Cluster Size", f"{df['size'].mean():.1f}")

# ---------- Raw data table (optional) ----------
with st.expander("� Raw Data Table"):
    st.dataframe(
        df[['cluster_id', 'label', 'size']].sort_values('size', ascending=False),
        use_container_width=True
    )
