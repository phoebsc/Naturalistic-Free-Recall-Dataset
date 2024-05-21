"""
semantic centrality
"""
import numpy as np
from scipy.spatial.distance import cdist
import more_itertools as mit
import os
import networkx as nx
import plotly.graph_objects as go
import sys
import kaleido
try:
    os.chdir('./topic_models')
except:
    pass
sys.path.append(os.getcwd())
import pandas as pd
from sentence_transformers import SentenceTransformer
from plotly.subplots import make_subplots

# load HMM info
data_dir = 'result_models'
txt_dir = 'result_text'
############################
n_topics = 40
story_size = 55
step_size = 21
"""
semantic network graphs
 Edge weights were thresholded at cosine similarity 0.6 for visualization purposes. Node size is proportional 
 to centrality computed from unthresholded networks. Edge thickness is proportional to edge weights.
"""
### code
def map_value(x, a, b, c, d):
    v = c + ((x - a) / (b - a)) * (d - c)
    return float("{:.2f}".format(max(min(d, v), c)))


# Assuming `embeddings` is your 25x768 numpy array
def compute_cosine_similarity(embeddings):
    norm = np.linalg.norm(embeddings, axis=1)
    norm[norm == 0] = 1
    embeddings_normalized = embeddings / norm[:, np.newaxis]
    return np.dot(embeddings_normalized, embeddings_normalized.T)


# Create graph based on cosine similarity
def create_graph(cos_sim, threshold, mode='path'):
    G = nx.Graph()
    n = cos_sim.shape[0]
    # Add nodes with initial centrality values
    for i in range(n):
        np.fill_diagonal(cos_sim, np.nan)
        G.add_node(i, centrality=np.nanmean(cos_sim[i]))  # Calculate unthresholded centrality
    if mode == 'path':
        for i in range(n - 1):  # Avoid out-of-index by stopping at n-1
            if cos_sim[i, i+1] > threshold:
                G.add_edge(i, i + 1, weight=cos_sim[i, i + 1])
    else:
        # Add edges with weights if cosine similarity is above threshold
        for i in range(n):
            for j in range(i + 1, n):
                # if cos_sim[i, j] > threshold:
                G.add_edge(i, j, weight=cos_sim[i, j])
    min_centrality = min(G.nodes[node]['centrality'] for node in G)
    max_centrality = max(G.nodes[node]['centrality'] for node in G)
    print(min_centrality,max_centrality)
    edge_weights = [G[u][v]['weight'] for u, v in G.edges()]

    # Calculate min and max
    # min_weight = min(edge_weights)
    # max_weight = max(edge_weights)
    # print(min_weight, max_weight)
    return G


# Plot using plotly
def plot_graph(G, color, pos, connect_lower, connect_upper,centrality_lower, centrality_upper, show_colorbar=False):
    # Plot using plotly
    edge_traces = []
    for edge in G.edges():
        x0, y0 = pos[edge[0]]
        x1, y1 = pos[edge[1]]
        weight = G[edge[0]][edge[1]]['weight']
        transparency = map_value(weight, connect_lower, connect_upper, 0, 1)
        edge_trace = go.Scatter(
            x=[x0, x1, None],
            y=[y0, y1, None],
            line=dict(width=2, color=f'rgba(50, 50, 50, {transparency})'),
            mode='lines')
        edge_traces.append(edge_trace)

    node_x = [pos[i][0] for i in G.nodes()]
    node_y = [pos[i][1] for i in G.nodes()]
    node_trace = go.Scatter(
        x=node_x, y=node_y,
        mode='markers',
        hoverinfo='text',
        marker=dict(
            showscale=show_colorbar,  # This controls whether to show the colorbar
            size=[map_value(G.nodes[n]['centrality'], centrality_lower, centrality_upper, 7, 17) for n in
                  G.nodes()],
            color=color,
            opacity=1,  # Make markers solid
            colorbar=dict(thickness=15, title='Centrality', xanchor='left', titleside='right')
        )
    )

    fig = go.Figure(data=edge_traces + [node_trace],
                    layout=go.Layout(
                        title='Network Graph based on Cosine Similarity',
                        showlegend=False,
                        hovermode='closest',
                        margin=dict(b=0, l=0, r=0, t=0),
                        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                        paper_bgcolor='white',  # White background
                        plot_bgcolor='white'  # White background for plot area
                    ))
    return fig

# Main function to process the embeddings and plot the graph
def main(embedding,subfolder, color, show_colorbar, mode):
    connect_lower, connect_upper, centrality_lower, centrality_upper = THRESHOLD, 0.75, 0.11, 0.44
    cos_sim = compute_cosine_similarity(embedding)
    np.save(os.path.join(data_dir,subfolder,'sent_embedding_sim'),  np.array(cos_sim))
    G = create_graph(cos_sim, connect_lower, mode)
    pos = nx.spring_layout(G, seed=42)  # Node positions
    return plot_graph(G, color, pos, connect_lower, connect_upper, centrality_lower, centrality_upper, show_colorbar)  # min and max across all 4 stories


# Assuming embeddings are loaded
figs = []
story_ids = ['pieman','eyespy','oregon','baseball']
mode = 'netwrk'
THRESHOLD = 0.4
colors = ['#fc8d62','#66c2a5','#e78ac3','#8da0cb']
for index,story_id in enumerate(story_ids):
    show_colorbar = (index == len(story_ids) - 1)
    subfolder = f'{story_id}_t{n_topics}_v{story_size}_r{story_size}_s{step_size}'
    f = open(os.path.join(txt_dir,subfolder,'story.txt'), 'r')
    lines = f.readlines()
    lines = [l.split('. ')[1].replace('\n','') for l in lines if len(l)>40]  # exclude super short sentences
    if story_id == 'pieman':
        lines = lines[0:-1]
    model = SentenceTransformer('all-mpnet-base-v2')
    embedding = model.encode(lines)
    figs.append(main(embedding,subfolder, colors[index], show_colorbar, mode))

# Create a 2x2 subplot layout
story_names = ['pieman','eyespy','oregontrail','baseball']
fig = make_subplots(rows=1, cols=4, subplot_titles=story_names,
                    horizontal_spacing=0.05, vertical_spacing=0.1)  # Adjust spacing as needed
# Add traces from each figure to the subplot
for i, f in enumerate(figs, start=1):
    for trace in f.data:
        fig.add_trace(trace, row=1, col=i)

# Update layout if necessary
fig.update_layout(
    title='',
    paper_bgcolor='white',
    plot_bgcolor='white',
    showlegend=False,
    height=400, width=1500,
    xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
    yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
    margin=dict(t=50, l=0, r=0, b=0)  # Adjust margins to tighten layout
)
fig.write_image("network_graph%s.svg" % THRESHOLD)

"""
generate an excel for LMM (analysis is done in R)
"""
story_ids = ['pieman','eyespy','oregon','baseball']
df_all = pd.DataFrame(columns=['event_id','story_id','participant_id','centrality','recalled'])
n=0
for story_id in story_ids:
    subfolder = '%s_t40_v55_r55_s21' % story_id
    filename = [x for x in os.listdir(os.path.join(data_dir, subfolder)) if 'precision_array' in x][0]
    _, _, recall_ids = np.load(os.path.join(os.getcwd(), data_dir, subfolder + '.npy'),
                                                     allow_pickle=True)
    precisions = np.load(os.path.join(data_dir, subfolder, filename), allow_pickle=True)
    precisions[precisions > 0] = 1  # turn precision matrix into probability of recall
    # load cos sim
    filename = [x for x in os.listdir(os.path.join(data_dir, subfolder)) if 'sent_embedding_sim' in x][0]
    cos_sim = np.load(os.path.join(data_dir, subfolder, filename), allow_pickle=True)
    np.fill_diagonal(cos_sim, np.nan)
    centrality = np.nanmean(cos_sim, axis=0)
    # make them the same length
    precisions = precisions[:,0:len(centrality)]
    for prec,name in zip(precisions, recall_ids):
        try:
            par_id = os.path.basename(name).split('_')[0]
            for ev, p in enumerate(prec):
                df_all.loc[n] = [ev, story_id, par_id, centrality[ev], p]
                n+=1
        except:
            pass

df_all.to_excel('LMM_all.xlsx')


