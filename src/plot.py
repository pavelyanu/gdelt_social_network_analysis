from typing import Dict, Tuple

import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
import plotly.express as px
import plotly.graph_objects as go

def plot_gdelt_network(G: nx.Graph) -> None:
    """
    Plots the GDELT network.
    """
    # clear old figure
    plt.clf
    pos = nx.spring_layout(G)
    nx.draw(G, pos, with_labels=True, node_size=20, font_size=8)
    plt.show()
    

def plot_top_centrality_nodes(df: pd.DataFrame, centrality_measure: str, top_n: int = 10) -> None:
    """
    Plot a bar chart of the top N nodes for a given centrality measure.
    df: DataFrame with centrality scores (index are nodes).
    """
    if centrality_measure not in df.columns:
        raise ValueError(f"{centrality_measure} not found in DataFrame. Available columns: {df.columns}")
    top_nodes = df[centrality_measure].dropna().sort_values(ascending=False).head(top_n)
    plt.figure(figsize=(10, 6))
    top_nodes.plot.barh()
    plt.title(f"Top {top_n} by {centrality_measure}")
    plt.gca().invert_yaxis()  # highest at top
    plt.xlabel(centrality_measure)
    plt.show()


def plot_communities(G: nx.Graph, communities: pd.Series) -> None:
    """
    Color nodes by community and display the network.
    """
    pos = nx.spring_layout(G, seed=42)  # fixed layout for consistency

    unique_comms = communities.unique()
    color_map = plt.cm.get_cmap('hsv', len(unique_comms))
    node_colors = [color_map(c) for c in communities]

    plt.figure(figsize=(10,10))
    nx.draw_networkx_nodes(G, pos, node_color=node_colors, node_size=50)
    nx.draw_networkx_edges(G, pos, alpha=0.3)
    # If labels needed:
    nx.draw_networkx_labels(G, pos, font_size=6)
    plt.title("Communities")
    plt.axis('off')
    plt.show()


def plot_network_summary_over_time(summary_df: pd.DataFrame) -> None:
    """
    Given a summary DataFrame with columns like year, density, etc.,
    plot how these metrics evolve over time.
    """
    if 'year' not in summary_df.columns:
        raise ValueError("summary_df must have a 'year' column")

    metrics = [col for col in summary_df.columns if col not in ['year']]
    summary_df = summary_df.set_index('year')

    summary_df[metrics].plot(subplots=True, layout=(len(metrics),1), figsize=(10, 6), sharex=True)
    plt.tight_layout()
    plt.show()


def plot_network_with_year_slider(networks: Dict[int, nx.Graph]):
    years = sorted(networks.keys())
    initial_year = years[0]
    G = networks[initial_year]

    fig, ax = plt.subplots(figsize=(10, 8))
    plt.subplots_adjust(bottom=0.2)

    pos = nx.spring_layout(G, seed=42)

    nodes = nx.draw_networkx_nodes(G, pos, ax=ax, node_size=50)
    edges = nx.draw_networkx_edges(G, pos, ax=ax, alpha=0.3)
    labels = nx.draw_networkx_labels(G, pos, font_size=6, ax=ax)
    ax.set_title(f"Network for Year {initial_year}")
    ax.axis('off')

    metric_ax = fig.add_axes([0.85, 0.4, 0.13, 0.5])
    metric_ax.axis('off')

    def update_metrics(G_current, year):
        metric_ax.clear()
        metric_ax.axis('off')
        num_nodes = G_current.number_of_nodes()
        num_edges = G_current.number_of_edges()
        density = nx.density(G_current)
        metric_ax.text(0, 0.8, f"Year: {year}", fontsize=10)
        metric_ax.text(0, 0.6, f"Nodes: {num_nodes}", fontsize=10)
        metric_ax.text(0, 0.4, f"Edges: {num_edges}", fontsize=10)
        metric_ax.text(0, 0.2, f"Density: {density:.4f}", fontsize=10)

    update_metrics(G, initial_year)

    slider_ax = fig.add_axes([0.2, 0.05, 0.6, 0.03])
    year_slider = Slider(slider_ax, 'Year', years[0], years[-1], valinit=initial_year, valstep=1)

    def update(val):
        year = int(year_slider.val)
        if year in networks:
            G_new = networks[year]
            ax.clear()
            ax.axis('off')
            pos_new = nx.spring_layout(G_new, seed=42)
            nx.draw_networkx_nodes(G_new, pos_new, ax=ax, node_size=50)
            nx.draw_networkx_edges(G_new, pos_new, ax=ax, alpha=0.3)
            nx.draw_networkx_labels(G_new, pos_new, font_size=6, ax=ax)
            ax.set_title(f"Network for Year {year}")
            update_metrics(G_new, year)
            fig.canvas.draw_idle()

    year_slider.on_changed(update)

    plt.show()
    
def plot_top_k_centrality_by_year(centralities_per_year: Dict[int, pd.DataFrame], centrality_measure: str, top_k: int = 10):
    """
    Plot top k nodes by a given centrality measure with a year slider.
    networks: Dict[int, nx.Graph]
    centrality_measure: str, one of 'degree_centrality', 'betweenness', 'closeness', 'eigenvector'
    top_k: how many top nodes to show
    """

    years = sorted(centralities_per_year.keys())
    initial_year = years[0]

    fig, ax = plt.subplots(figsize=(8, 6))
    plt.subplots_adjust(bottom=0.25)

    def plot_year(yr):
        df = centralities_per_year[yr]
        if centrality_measure not in df.columns:
            raise ValueError(f"Invalid centrality measure. Available: {df.columns}")

        top_nodes = df[centrality_measure].dropna().sort_values(ascending=False).head(top_k)
        ax.clear()
        top_nodes.plot.barh(ax=ax)
        ax.set_title(f"Top {top_k} by {centrality_measure} - Year {yr}")
        ax.set_xlabel(centrality_measure)
        ax.invert_yaxis()

    plot_year(initial_year)

    slider_ax = fig.add_axes([0.2, 0.1, 0.6, 0.03])
    year_slider = Slider(slider_ax, 'Year', years[0], years[-1], valinit=initial_year, valstep=1)

    def update(val):
        yr = int(year_slider.val)
        if yr in centralities_per_year:
            plot_year(yr)
            fig.canvas.draw_idle()

    year_slider.on_changed(update)
    plt.show()
    
def align_communities(communities_by_year: Dict[int, pd.DataFrame]) -> Dict[int, pd.DataFrame]:
    years = sorted(communities_by_year.keys())
    ref_year = years[0]

    # Sort reference year's communities by community ID and use them as a stable baseline
    ref_df = communities_by_year[ref_year].copy()
    # Suppose communities are integer-labeled. Sort them by something stable, like their mean ISO code or just their ID.
    # For simplicity, assume they are already labeled 0,1,2,...
    # If not, you might want to reorder them by size or something consistent.
    ref_df['old_label'] = ref_df['community']  # Just keep as is
    communities_by_year[ref_year] = ref_df

    previous_year_df = ref_df

    for yr in years[1:]:
        current_df = communities_by_year[yr].copy()

        # Find unique communities this year
        curr_comms = current_df['community'].unique()
        # Prepare a mapping from new community -> old community label
        new_label_map = {}

        # For each new community, find the old community it overlaps with most
        for c in curr_comms:
            curr_nodes = set(current_df[current_df['community'] == c]['iso'])
            best_overlap = 0
            best_old_label = None

            # Check overlap with old communities from previous_year_df
            for old_c in previous_year_df['community'].unique():
                old_nodes = set(previous_year_df[previous_year_df['community'] == old_c]['iso'])
                overlap = len(curr_nodes & old_nodes)
                if overlap > best_overlap:
                    best_overlap = overlap
                    best_old_label = old_c

            # If no overlap or you want some threshold, handle it here:
            if best_old_label is None:
                # Assign a new label. For simplicity, just use the same label c or max + 1
                # Or you could track new labels across years.
                best_old_label = c

            new_label_map[c] = best_old_label

        # Apply mapping
        current_df['community'] = current_df['community'].map(new_label_map)

        # Update previous_year_df for next iteration
        previous_year_df = current_df
        communities_by_year[yr] = current_df

    return communities_by_year


def plot_communities_on_map(communities_dict: Dict[int, pd.DataFrame]) -> go.Figure:
    """
    Creates a choropleth map with a slider to show communities by year.
    communities_dict: Dict[int, pd.DataFrame]
        Each DataFrame should have columns ['iso', 'country_name', 'community'].
    """
    communities_dict = align_communities(communities_dict)
    all_records = []
    for year, df in communities_dict.items():
        temp = df.copy()
        temp['year'] = year
        temp['iso'] = temp['iso'].str.upper()
        all_records.append(temp)

    combined_df = pd.concat(all_records, ignore_index=True)

    combined_df['community'] = combined_df['community'].astype(str)

    fig = px.choropleth(
        combined_df,
        locations='iso',
        color='community',
        hover_name='country_name',
        animation_frame='year',
        projection='natural earth',
        title='Communities Over Years',
        color_discrete_sequence=px.colors.qualitative.Safe
    )

    fig.update_layout(
        width=1000,
        height=600,
        coloraxis_colorbar=dict(
            title='Community',
            ticks='outside',
            ticklen=5,
            showticksuffix='last'
        )
    )

    return fig

def plot_network_on_map_with_slider(gdelt_df: pd.DataFrame, centroids: Dict[str, Tuple[float, float]]) -> go.Figure:
    """
    Plot the GDELT network on a world map with a slider for years.
    Each edge (country pair) is a line whose color is determined by average tone
    and whose opacity is determined by sum_nummentions, both normalized within the year.
    """

    gdelt_df['actor1country'] = gdelt_df['actor1country'].str.upper()
    gdelt_df['actor2country'] = gdelt_df['actor2country'].str.upper()

    years = sorted(gdelt_df['year'].unique())
    frames = []
    initial_data = []

    color_scale = px.colors.diverging.RdYlGn

    def value_to_color(v, vmin, vmax):
        t = (v - vmin) / (max(vmax - vmin, 1e-9))
        idx = int(t * (len(color_scale)-1))
        return color_scale[idx]

    def value_to_opacity(v, vmin, vmax):
        t = (v - vmin) / (max(vmax - vmin, 1e-9))
        return 0.4 + 0.6 * t

    for i, yr in enumerate(years):
        df_yr = gdelt_df[gdelt_df['year'] == yr].copy()

        tone_min, tone_max = df_yr['weighted_sum_avgtone'].min(), df_yr['weighted_sum_avgtone'].max()
        num_min, num_max = df_yr['sum_nummentions'].min(), df_yr['sum_nummentions'].max()

        traces = []
        for row in df_yr.itertuples(index=False):
            iso1 = row.actor1country
            iso2 = row.actor2country
            if iso1 not in centroids or iso2 not in centroids:
                continue

            lat1, lon1 = centroids[iso1]
            lat2, lon2 = centroids[iso2]

            avgtone = row.weighted_sum_avgtone
            num = row.sum_nummentions

            color = value_to_color(avgtone, tone_min, tone_max)
            opacity = value_to_opacity(num, num_min, num_max)

            line_trace = go.Scattergeo(
                lat=[lat1, lat2],
                lon=[lon1, lon2],
                mode='lines',
                line=dict(color=color, width=2),
                opacity=opacity,
                showlegend=False,
                hovertext=f"{row.actor1country_name} -> {row.actor2country_name}<br>",
            )
            traces.append(line_trace)

        frame = go.Frame(data=traces, name=str(yr))
        frames.append(frame)
        if i == 0:
            initial_data = traces

    fig = go.Figure(data=initial_data, frames=frames)

    steps = []
    for i, yr in enumerate(years):
        step = dict(
            method='animate',
            args=[[str(yr)],
                  dict(mode='immediate', frame=dict(duration=500, redraw=True), transition=dict(duration=0))],
            label=str(yr)
        )
        steps.append(step)

    sliders = [dict(
        active=0,
        currentvalue={"prefix": "Year: "},
        steps=steps
    )]

    fig.update_layout(
        title='GDELT Network Over Years',
        geo=dict(scope='world', projection=dict(type='natural earth')),
        sliders=sliders,
        updatemenus=[dict(
            type='buttons',
            showactive=False,
            y=1,
            x=1.1,
            xanchor='left',
            yanchor='top',
            buttons=[dict(label='Play', method='animate', args=[None])]
        )],
        width=1200,
        height=800
    )

    fig.update_geos(
        showland=True,
        landcolor="lightgray",
        showocean=True,
        oceancolor="aliceblue",
        showcoastlines=True,
        coastlinecolor="black",
        coastlinewidth=0.5,
        showcountries=True,
        countrycolor="black",
        countrywidth=0.5,
        projection_type="natural earth"
    )

    return fig
