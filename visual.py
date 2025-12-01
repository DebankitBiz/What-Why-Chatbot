
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go

def visual(df_result,x_axis,y_axis,color_by,secondary_y_axis,chart_type,chart_title,
           color_scheme,trendline_flag,regression_type,COLOR_QUAL,COLOR_CONT,make_subplots):
    #st.write("Visualization Invoke")
    if df_result is not None and not df_result.empty:
        #st.dataframe(df_result, use_container_width=True)

        # ADVANCED VISUALIZATION ENGINE (PATCHED)
        # =========================================
        st.write("### 📈 Visualization")

        if df_result is not None and not df_result.empty:

            df_sorted = df_result.copy()
            cols = list(df_sorted.columns)
            cols = [str(c) for c in cols]
            
            numeric_cols = df_sorted.select_dtypes(include="number").columns.tolist()
            cat_cols = [c for c in cols if c not in numeric_cols]

            # -------------------------------
            # Safe axis selection (LLM repair)
            # -------------------------------
            x_col = x_axis if x_axis in cols else None
            y_col = y_axis if y_axis in cols else None
            color_col = color_by if color_by in cols else None
            sec_y_col = secondary_y_axis if secondary_y_axis in cols else None

            # AUTO-REPAIR X AXIS
            if x_col is None:
                if len(cat_cols) > 0:
                    x_col = cat_cols[0]
                else:
                    # fallback: any non-constant column
                    non_constant = [c for c in cols if df_sorted[c].nunique() > 1]
                    x_col = non_constant[0] if non_constant else cols[0]

            # Fix if x_col is constant
            if df_sorted[x_col].nunique() <= 1:
                # pick next best category
                fallback_x = [c for c in cat_cols if df_sorted[c].nunique() > 1]
                if fallback_x:
                    x_col = fallback_x[0]
                else:
                    # pick any variable column
                    variable_cols = [c for c in cols if df_sorted[c].nunique() > 1]
                    if variable_cols:
                        x_col = variable_cols[0]

            # AUTO-REPAIR Y AXIS
            if y_col not in numeric_cols:
                if len(numeric_cols) > 0:
                    y_col = numeric_cols[0]
                else:
                    st.info("No numeric column available for visualization.")
                    chart_type = "none"

            # -----------------------------------
            # Intelligent automatic sorting (Q1, Months, W1)
            # -----------------------------------

            def sort_if_sequence(df, col):
                vals = df[col].astype(str).tolist()

                # Quarter Q1–Q4
                if all(v.startswith("Q") and v[1:].isdigit() for v in vals):
                    df["__sort_col"] = df[col].str.extract(r"Q(\d+)").astype(int)
                    df = df.sort_values("__sort_col")
                    return df.drop(columns="__sort_col")

                # Week W1–W52
                if all(v.startswith("W") and v[1:].isdigit() for v in vals):
                    df["__sort_col"] = df[col].str.extract(r"W(\d+)").astype(int)
                    df = df.sort_values("__sort_col")
                    return df.drop(columns="__sort_col")

                # Month names
                months = {
                    "January":1,"February":2,"March":3,"April":4,"May":5,"June":6,
                    "July":7,"August":8,"September":9,"October":10,"November":11,"December":12
                }
                if all(v in months for v in vals):
                    df["__sort_col"] = df[col].map(months)
                    df = df.sort_values("__sort_col")
                    return df.drop(columns="__sort_col")

                return df

            df_sorted = sort_if_sequence(df_sorted, x_col)

            # -----------------------------------
            # Choose valid trendline (scatter only)
            # -----------------------------------
            trendline_arg = None
            if trendline_flag and regression_type in ["ols", "lowess"]:
                if chart_type in ["scatter", "bubble"]:
                    trendline_arg = regression_type

            # -----------------------------------
            # Build figure
            # -----------------------------------
            fig = None
            chart_title_final = (chart_title or "Chart").title().strip()
            # st.write(chart_type)
            # st.write(x_col)
            # st.write(y_col)
            #2-COLUMN CHARTS
            if len(cols) == 2 or  chart_type in ["bar", "line", "area", "scatter", "pie", "histogram", "box", "violin"]:

                try:
                    if chart_type == "bar":
                        fig = px.bar(df_sorted, x=x_col, y=y_col, title=chart_title_final,
                                    color_discrete_sequence=COLOR_QUAL.get(color_scheme))

                    elif chart_type == "line":
                        fig = px.line(df_sorted, x=x_col, y=y_col, markers=True, title=chart_title_final,
                                    color_discrete_sequence=COLOR_QUAL.get(color_scheme))

                    elif chart_type == "area":
                        fig = px.area(df_sorted, x=x_col, y=y_col, title=chart_title_final)

                    elif chart_type == "scatter":
                        fig = px.scatter(df_sorted, x=x_col, y=y_col, title=chart_title_final,
                                        color_discrete_sequence=COLOR_QUAL.get(color_scheme),
                                        trendline=trendline_arg)

                    elif chart_type == "pie":
                        fig = px.pie(df_sorted, names=x_col, values=y_col, title=chart_title_final,
                                    color_discrete_sequence=COLOR_QUAL.get(color_scheme))

                    elif chart_type == "histogram":
                        fig = px.histogram(df_sorted, x=y_col if y_col in numeric_cols else x_col,
                                        title=chart_title_final,
                                        color_discrete_sequence=COLOR_QUAL.get(color_scheme))

                    elif chart_type == "box":
                        fig = px.box(df_sorted, x=x_col, y=y_col, title=chart_title_final,
                                    color_discrete_sequence=COLOR_QUAL.get(color_scheme))

                    elif chart_type == "violin":
                        fig = px.violin(df_sorted, x=x_col, y=y_col, box=True, points="all",
                                        title=chart_title_final,
                                        color_discrete_sequence=COLOR_QUAL.get(color_scheme))
                except Exception as e:
                    st.error(f"Visualization error: {e}")

            # 3-COLUMN CHARTS
            elif len(cols) == 3 and chart_type in ["grouped_bar", "stacked_bar", "treemap", "sunburst", "heatmap", "bubble", "histogram", "box", "violin"]:

                dim1, dim2, metric = cols

                if metric not in numeric_cols:
                    for c in cols:
                        if c in numeric_cols:
                            metric = c
                            break

                try:
                    if chart_type == "grouped_bar":
                        fig = px.bar(df_sorted, x=dim1, y=metric, color=dim2, barmode="group",
                                    title=chart_title_final,
                                    color_discrete_sequence=COLOR_QUAL.get(color_scheme))

                    elif chart_type == "stacked_bar":
                        fig = px.bar(df_sorted, x=dim1, y=metric, color=dim2, barmode="relative",
                                    title=chart_title_final,
                                    color_discrete_sequence=COLOR_QUAL.get(color_scheme))

                    elif chart_type == "treemap":
                        fig = px.treemap(df_sorted, path=[dim1, dim2], values=metric,
                                        title=chart_title_final,
                                        color=dim2,
                                        color_continuous_scale=COLOR_CONT.get(color_scheme))

                    elif chart_type == "sunburst":
                        fig = px.sunburst(df_sorted, path=[dim1, dim2], values=metric,
                                        title=chart_title_final,
                                        color=dim2,
                                        color_continuous_scale=COLOR_CONT.get(color_scheme))

                    elif chart_type == "heatmap":
                        fig = px.density_heatmap(df_sorted, x=dim1, y=dim2, z=metric,
                                                title=chart_title_final,
                                                color_continuous_scale=COLOR_CONT.get(color_scheme))

                    elif chart_type == "bubble":
                        fig = px.scatter(df_sorted, x=dim1, y=metric, size=metric,
                                        color=dim2, title=chart_title_final,
                                        color_discrete_sequence=COLOR_QUAL.get(color_scheme),
                                        trendline=trendline_arg)

                    elif chart_type == "histogram":
                        fig = px.histogram(df_sorted, x=metric, color=dim1,
                                        title=chart_title_final,
                                        color_discrete_sequence=COLOR_QUAL.get(color_scheme))

                    elif chart_type == "box":
                        fig = px.box(df_sorted, x=dim1, y=metric, color=dim2,
                                    title=chart_title_final,
                                    color_discrete_sequence=COLOR_QUAL.get(color_scheme))

                    elif chart_type == "violin":
                        fig = px.violin(df_sorted, x=dim1, y=metric, color=dim2,
                                        box=True, points="all",
                                        title=chart_title_final,
                                        color_discrete_sequence=COLOR_QUAL.get(color_scheme))

                except Exception as e:
                    st.error(f"Visualization error: {e}")

            # DUAL-AXIS CHARTS
            elif chart_type in ["dual_axis_line", "dual_axis_bar"] and sec_y_col and sec_y_col in numeric_cols:
                try:
                    fig = make_subplots(specs=[[{"secondary_y": True}]])
                    x_vals = df_sorted[x_col]

                    if chart_type == "dual_axis_bar":
                        fig.add_trace(go.Bar(x=x_vals, y=df_sorted[y_col], name=y_col), secondary_y=False)
                    else:
                        fig.add_trace(go.Scatter(x=x_vals, y=df_sorted[y_col], name=y_col, mode="lines+markers"), secondary_y=False)

                    fig.add_trace(go.Scatter(x=x_vals, y=df_sorted[sec_y_col], name=sec_y_col, mode="lines+markers"), secondary_y=True)
                    fig.update_layout(title=chart_title_final)

                except Exception as e:
                    st.error(f"Visualization error: {e}")

            # HIGH-DIM SCATTER MATRIX
            elif len(cols) >= 4 and len(numeric_cols) >= 2:
                try:
                    fig = px.scatter_matrix(df_sorted[numeric_cols], title=chart_title_final)
                except Exception as e:
                    st.error(f"Visualization error: {e}")

            # Render
            if fig:
                fig.update_layout(title={"text": chart_title_final, "x": 0.5, "xanchor": "center", "font": {"size": 24}})
                #st.plotly_chart(fig, use_container_width=True)
                return fig 
            else:
                st.info("No meaningful chart could be generated.")
                return None
    else:
        st.info("No rows returned.")
        return None


        

def plot_combined_rca(rca_results):
    """
    Plots the top contributor from each dimension in a clean readable format.
    Sorted by delta (pct_recent - pct_history) in decreasing order.
    """

    rows = []  # temporary list to sort later

    for dim_name, df in rca_results.items():
        if df.empty:
            continue

        top = df.iloc[0]

        # Identify dimension columns (everything except last 3 metric columns)
        dim_cols = df.columns[:len(df.columns)-3]

        # Determine readable dimension label
        if len(dim_cols) == 1:
            readable_dim = dim_cols[0]
        elif len(dim_cols) == 2:
            readable_dim = f"{dim_cols[0]} and {dim_cols[1]}"
        elif len(dim_cols) == 3:
            readable_dim = f"{dim_cols[0]}, {dim_cols[1]} and {dim_cols[2]}"
        else:
            readable_dim = dim_name

        # Build readable item text
        if len(dim_cols) == 1:
            item_text = str(top[dim_cols[0]])
        else:
            item_text = " | ".join(str(top[c]) for c in dim_cols)

        final_label = f"{readable_dim} → {item_text}"

        rows.append({
            "label": final_label,
            "spike": top["pct_recent"],
            "hist": top["pct_history"],
            "delta": abs(top["pct_recent"] - top["pct_history"])
        })

    # Sort by delta descending
    sorted_rows = sorted(rows, key=lambda x: x["delta"], reverse=True)

    # Extract sorted lists
    labels = [r["label"] for r in sorted_rows]
    spike_vals = [r["spike"] for r in sorted_rows]
    hist_vals  = [r["hist"] for r in sorted_rows]

    # Plotly chart
    fig = go.Figure()

    fig.add_trace(go.Bar(
        x=labels,
        y=spike_vals,
        name="Specified Period",
        marker_color="#1f77b4"
    ))

    fig.add_trace(go.Bar(
        x=labels,
        y=hist_vals,
        name="Historical Avg",
        marker_color="#c7c7c7"
    ))

    fig.update_layout(
        title="Top Contributors Across All Dimensions",
        xaxis_title="Driver",
        yaxis_title="% Contribution",
        barmode="group",
        height=520,
        xaxis_tickangle=-25,
        margin=dict(l=40, r=40, t=60, b=150)
    )

    return fig
