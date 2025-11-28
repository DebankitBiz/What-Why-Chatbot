import plotly.graph_objects as go

def plot_combined_rca(rca_results):
    """
    rca_results = {
        "Product Name": df,
        "Region": df,
        "Product Class": df,
        "Sales Team": df
    }

    Each df must include:
    - dim_col   (first column: the category label)
    - pct_recent
    - pct_history
    """

    dim_labels = []
    spike_vals = []
    history_vals = []

    for dim_name, df in rca_results.items():
        if df.empty:
            continue
        
        # top driver is first row (already sorted)
        top = df.iloc[0]
        label = f"{top.iloc[0]} ({dim_name})"

        dim_labels.append(label)
        spike_vals.append(top["pct_recent"])
        history_vals.append(top["pct_history"])

    fig = go.Figure()

    # Spike period (Blue)
    fig.add_trace(go.Bar(
        x=dim_labels,
        y=spike_vals,
        name="Spike Period",
        marker_color="royalblue"
    ))

    # Historical Avg (Gray)
    fig.add_trace(go.Bar(
        x=dim_labels,
        y=history_vals,
        name="Historical Avg",
        marker_color="lightgray"
    ))

    # Layout
    fig.update_layout(
        title="Combined RCA Contribution Analysis",
        xaxis_title="Dimensions (Top Driver)",
        yaxis_title="% Contribution to Metric",
        barmode="group",
        height=500,
        margin=dict(l=40, r=40, t=60, b=120),
        xaxis_tickangle=-30
    )

    return fig
