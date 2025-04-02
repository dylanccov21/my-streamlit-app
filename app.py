import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
import plotly.express as px
import tensorflow as ts
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense

st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", ["Home", "Storage & Handling", "Freight","ML Test"])

# ✅ Declare file uploader first to avoid NameError
uploaded_file = st.sidebar.file_uploader("Upload an Excel file", type=["xlsx", "xls"])

if uploaded_file:
    try:
        # Load Excel file
        xls = pd.ExcelFile(uploaded_file)
        sheet_names = xls.sheet_names

    except Exception as e:
        st.sidebar.error(f"Error loading file: {e}")
        st.stop()


# ✅ Page 1: Home (Displays Hardcoded Images)
if page == "Home":
    st.title("SCCT4.0 Dashboard")



    # ✅ Corrected Image Path (Use raw string or double backslashes)
    image_paths = [
        r"Implementation_Phase.png",
        r"KeyProjectTasks.png"# ✅ Fixed path
    ]

    # ✅ Display images using `use_container_width`
    for img in image_paths:
        st.image(img, use_container_width=True)  # ✅ Replaced `use_column_width`



# ✅ Page 2: Storage & Handling (Keeps Sidebar Sheet Selection)
elif page == "Storage & Handling":
    st.title("Storage & Handling")

    if uploaded_file:
        try:
            # ✅ Only show sheet selection in Storage & Handling
            default_sheet = "Total Result"
            excluded_sheets = ["GSC - Freight"]

            if default_sheet in excluded_sheets:
                st.error(f'The default sheet "{default_sheet}" is excluded. Please check the configuration.')
                st.stop()

            # You can still validate if the sheet exists in the uploaded Excel file (optional)
            if default_sheet not in sheet_names:
                st.error(f'Sheet "{default_sheet}" not found in the Excel file.')
                st.stop()

            selected_sheet = default_sheet
            # 🔹 Load the selected sheet
            df_main = pd.read_excel(xls, sheet_name=selected_sheet, header=0, dtype=str)
            df_main.columns = df_main.columns.astype(str)

            # 🔹 Continue with Storage & Handling processing...
            # 🔹 Ensure the dataframe has enough columns
            total_columns = len(df_main.columns)
            if total_columns < 116:
                st.error(f"Expected at least 116 columns, but found {total_columns}. Please check the file format.")
                st.stop()

            # 🔹 Select specific columns for analysis
            selected_columns = [10] + list(range(109, 115)) + [104, 105, 95, 13, 17]
            df_selected = df_main.iloc[:, selected_columns].copy()
            # Optional: Clean column 11 (index 10) more gently
            region_column = df_selected.columns[0]
            df_selected[region_column] = df_selected[region_column].fillna("NA")

            # 🔹 Convert CR column (Column 95) to numeric
            cr_column = df_selected.columns[9]
            df_selected[cr_column] = pd.to_numeric(df_selected[cr_column], errors='coerce').fillna(0)



            # 🔹 Sidebar Filters
            st.sidebar.title("Filters")
            filters = {}

            with st.sidebar.expander("🔽 Filter Displayed Data", expanded=True):
                for column in df_selected.columns:
                    unique_values = df_selected[column].dropna().unique()

                    # Filter for Year column
                    if "year" in column.lower():
                        selected_year = st.selectbox(f"Filter by {column}", ["All"] + sorted(map(str, unique_values)))
                        if selected_year != "All":
                            filters[column] = selected_year

                    # Numeric range filter
                    elif pd.to_numeric(df_selected[column], errors='coerce').notna().all():
                        df_selected[column] = pd.to_numeric(df_selected[column], errors='coerce')
                        min_val, max_val = df_selected[column].min(), df_selected[column].max()
                        if min_val != max_val:
                            selected_range = st.slider(f"{column} Range", float(min_val), float(max_val), (float(min_val), float(max_val)))
                            filters[column] = selected_range

                    # Categorical dropdown filter
                    elif len(unique_values) > 1:
                        selected_value = st.selectbox(f"Filter by {column}", ["All"] + sorted(map(str, unique_values)))
                        if selected_value != "All":
                            filters[column] = selected_value

            # 🔹 Apply filters dynamically
            for col, val in filters.items():
                if isinstance(val, tuple):  # Numeric range filter
                    df_selected = df_selected[(df_selected[col] >= val[0]) & (df_selected[col] <= val[1])]
                else:  # Categorical filter
                    df_selected = df_selected[df_selected[col].astype(str) == val]

            # 🔹 Display the filtered dataframe
            st.success(f"Showing filtered data from sheet: **{selected_sheet}**")
            st.dataframe(df_selected)

            # 🔹 Display the sum of CR (EUR)
            total_cr = df_selected[cr_column].sum()
            st.write("### Data Summary:")
            st.metric(label="Total Sum (EUR)", value=f"€{total_cr:,.2f}")

            import plotly.graph_objects as go

            # ✅ Load "GSC - Freight" data for the line chart
            df_freight = pd.read_excel(xls, sheet_name="GSC - Freight", header=2, dtype=str, keep_default_na=False)
            df_freight.columns = df_freight.columns.astype(str)

            # ✅ Convert Numeric Columns (J & L)
            col_j = df_freight.columns[9]  # Column J (Numeric Month)
            col_l = df_freight.columns[11]  # Column L (Total Volume Shipped)
            col_c = df_freight.columns[2]  # ✅ Column C (Region Plant)

            df_freight[col_j] = pd.to_numeric(df_freight[col_j], errors='coerce')
            df_freight[col_l] = pd.to_numeric(df_freight[col_l], errors='coerce').fillna(0)

            # ✅ Apply Sidebar Filters (A, B, C) - Auto-Filtering Column C
            freight_filters = {}

            with st.sidebar.expander("🔽 Filter Freight Data", expanded=True):
                for col in df_freight.columns[:3]:  # First 3 columns (A, B, C)
                    unique_values = df_freight[col].dropna().unique()

                    # ✅ Auto-filter Column C (Region Plant) based on selected sheet
                    if col == col_c:
                        if "APAC" in selected_sheet:
                            default_value = "APAC"
                        elif "NA" in selected_sheet:
                            default_value = "NA"
                        elif "EMEA" in selected_sheet:
                            default_value = "EMEA"
                        else:
                            default_value = "All"

                        selected_value = st.sidebar.selectbox(f"Filter by {col}",
                                                              ["All"] + sorted(map(str, unique_values)),
                                                              index=(["All"] + sorted(map(str, unique_values))).index(
                                                                  default_value))
                    else:
                        selected_value = st.sidebar.selectbox(f"Filter by {col}",
                                                              ["All"] + sorted(map(str, unique_values)))

                    if selected_value != "All":
                        freight_filters[col] = selected_value

            # ✅ Apply Filters to "GSC - Freight"
            for col, val in freight_filters.items():
                df_freight = df_freight[df_freight[col] == val]

            # ✅ Aggregate Line Chart Data (Sum of Column L Grouped by Numeric Month J)
            line_data = df_freight.groupby(col_j)[col_l].sum().sort_index()

            # ✅ Define Correct Month Order for Bar Chart
            month_column = df_selected.columns[10]  # Column 14 in original dataset
            month_order = [
                "January", "February", "March", "April", "May", "June",
                "July", "August", "September", "October", "November", "December"
            ]

            df_selected[month_column] = pd.Categorical(df_selected[month_column], categories=month_order, ordered=True)
            month_counts = df_selected[month_column].value_counts().reindex(month_order).fillna(0)

            # ✅ Fix Alignment Issue: Shift line data one position left
            line_data.index = [month_order[i - 1] for i in line_data.index if 1 <= i <= 12]

            # ✅ Create Multi-Type Chart with Dual Y-Axes (Independent Scaling)
            fig = go.Figure()

            # 🔹 Bar Chart (PO Line Items, Left Y-Axis)
            fig.add_trace(go.Bar(
                x=month_order,
                y=month_counts,
                name="Invoice Line Items",
                marker_color="skyblue",
                yaxis="y1"  # ✅ Left Y-Axis
            ))

            # ✅ Overlay Line Chart (Total Volume Shipped, Right Y-Axis)
            fig.add_trace(go.Scatter(
                x=month_order,
                y=line_data.reindex(month_order),
                mode="lines+markers",
                name="Total Volume Shipped",
                marker=dict(color="red", size=8),
                line=dict(color="red", width=2),
                hovertemplate="<b>%{x}</b><br>Total Volume Shipped: %{y:,.0f} KG",
                yaxis="y2"  # ✅ Right Y-Axis
            ))

            # ✅ Configure Independent Y-Axes
            fig.update_layout(
                title="Posting Date & Freight Volume",
                xaxis_title="Month",
                yaxis=dict(
                    title=dict(text="Invoice Line Items", font=dict(color="blue")),
                    tickfont=dict(color="blue"),
                    side="left",
                    showgrid=False  # ✅ Hide grid for better readability
                ),
                yaxis2=dict(
                    title=dict(text="Total Volume Shipped (KG)", font=dict(color="red")),
                    tickfont=dict(color="red"),
                    side="right",
                    overlaying="y",
                    showgrid=False  # ✅ Hide grid for better readability
                ),
                legend=dict(x=1.15, y=1),  # ✅ Move legend to right
                hovermode="x unified"  # ✅ Shows hover tooltip aligned on the x-axis
            )

            # 🔹 Display Chart in Streamlit
            st.plotly_chart(fig)

            # ✅ Restore Pie Chart (Under Bar Chart)
            sbe_column = df_selected.columns[7]
            sbe_counts = df_selected[sbe_column].value_counts()

            if not sbe_counts.empty:
                fig_pie, ax_pie = plt.subplots(figsize=(6, 6))
                ax_pie.pie(sbe_counts, labels=sbe_counts.index, autopct='%1.1f%%', colors=plt.cm.Paired.colors)
                ax_pie.set_title("PO Lines by SBE")
                st.pyplot(fig_pie)

                # ✅ Select necessary columns
                column_bn = df_main.columns[65]  # Column BN (Vendor)
                column_cr = df_main.columns[95]  # Column CR (Total Cost of Line Items)
                df_main[column_cr] = pd.to_numeric(df_main[column_cr], errors='coerce').fillna(0)

                # ✅ Sidebar Vendor Filters (Collapsible)
                with st.sidebar.expander("Vendor Filters", expanded=False):
                    unique_bn_values = df_main[column_bn].dropna().unique()
                    selected_bn = st.selectbox("Filter by BN", ["All"] + sorted(map(str, unique_bn_values)))

                    # ✅ Input field for filtering top percentage of total spend vendors
                    top_percentage = st.number_input("Enter Top % of Total Spend Vendors", min_value=1, max_value=100,
                                                     value=100, step=1)

                if selected_bn != "All":
                    df_main = df_main[df_main[column_bn] == selected_bn]

                # ✅ Bar Chart for Column BN with Total Cost (Sorted)
                bn_costs = df_main.groupby(column_bn)[column_cr].sum().sort_values(ascending=False)

                # ✅ Filter for top percentage of total spend vendors
                if not bn_costs.empty:
                    cumulative_sum = bn_costs.cumsum()
                    total_sum = bn_costs.sum()
                    cutoff_value = total_sum * (top_percentage / 100)
                    bn_costs = bn_costs[cumulative_sum <= cutoff_value]

                    fig_bar = px.bar(
                        x=bn_costs.index,
                        y=bn_costs.values,
                        title=f"Top {top_percentage}% Vendors by Total Spend",
                        labels={"x": "Vendor", "y": "Total Spend (EUR)"},
                        text_auto=True
                    )
                    fig_bar.update_layout(xaxis={'categoryorder': 'total descending'})
                    st.plotly_chart(fig_bar)

        except Exception as e:
            st.error(f"Error processing data: {e}")



# ✅ Page 3: Freight (No Sidebar Sheet Selection, Only Displays "GSC - Freight")
elif page == "Freight":
    st.title("Freight")
    st.write("Displaying the full 'GSC - Freight' dataset and visualizations.")

    if uploaded_file:
        try:
            # ✅ Explicitly Load "GSC - Freight" Sheet
            if "GSC - Freight" not in sheet_names:
                st.error("The 'GSC - Freight' sheet was not found in the uploaded file.")
                st.stop()

            df_gsc = pd.read_excel(xls, sheet_name="GSC - Freight", header=2, dtype=str, keep_default_na=False)
            df_gsc.columns = df_gsc.columns.astype(str)

            # ✅ Show currently displayed sheet in the sidebar (NO sheet selection)
            st.sidebar.title("Filters")
            st.sidebar.info("📂 **Currently Displaying:** GSC - Freight")

            # 🔹 Sidebar Filters (A, B, C, D)
            filters = {}
            filter_columns = df_gsc.columns[:4]

            with st.sidebar.expander("🔽 Filter Data", expanded=True):
                for col in filter_columns:
                    unique_values = df_gsc[col].dropna().unique()
                    selected_value = st.sidebar.selectbox(f"Filter by {col}", ["All"] + sorted(map(str, unique_values)))
                    if selected_value != "All":
                        filters[col] = selected_value

            # 🔹 Apply Filters Dynamically
            for col, val in filters.items():
                df_gsc = df_gsc[df_gsc[col] == val]

            # 🔹 Convert Numeric Columns (K & L)
            col_k = df_gsc.columns[10]
            col_l = df_gsc.columns[11]
            df_gsc[col_k] = pd.to_numeric(df_gsc[col_k], errors='coerce').fillna(0)
            df_gsc[col_l] = pd.to_numeric(df_gsc[col_l], errors='coerce').fillna(0)

            # 🔹 Display Full Dataset
            st.success(f"Displaying {df_gsc.shape[0]} rows and {df_gsc.shape[1]} columns from 'GSC - Freight'.")
            st.dataframe(df_gsc)

            # 🔹 Display "Total Freight Volume" & "Cost / KG"
            total_freight_volume = df_gsc[col_l].sum()
            total_kg = df_gsc[col_k].sum()
            cost_per_kg = total_freight_volume / total_kg if total_kg > 0 else 0.00

            st.subheader("Total Freight Metrics")
            st.metric(label="Total Freight Volume (KG)", value=f"{total_freight_volume:,.2f}")
            st.metric(label="Cost per KG", value=f"€{cost_per_kg:,.4f}")

            # 🔹 Pie Chart Toggle (Region vs. Plant)
            pie_option = st.radio("View Pie Chart by:", ["Region (C)", "Plant (D)"], horizontal=True)
            pie_column = df_gsc.columns[2] if pie_option == "Region (C)" else df_gsc.columns[3]
            pie_data = df_gsc.groupby(pie_column)[col_l].sum().dropna()

            if not pie_data.empty:
                fig, ax = plt.subplots(figsize=(7, 7))
                ax.pie(pie_data, labels=[f"{c}: {v:,.0f} KG" for c, v in pie_data.items()],
                       colors=plt.cm.Paired.colors, labeldistance=1.2)
                ax.set_title(f"Total Freight Volume by {pie_column}")
                st.pyplot(fig)

        except Exception as e:
            st.error(f"Error reading 'GSC - Freight' sheet: {e}")

if page == "ML Test":
    st.title("📊 Storage Cost Prediction using TensorFlow")

    if uploaded_file:
        try:
            selected_sheet = st.sidebar.selectbox("Select a sheet for cost prediction", sheet_names)
            df = pd.read_excel(uploaded_file, sheet_name=selected_sheet, header=2, dtype=str)

            st.write("### First 10 Rows of Uploaded Data (Raw)")
            st.dataframe(df.head(10))

            month_column = "Posting period"
            cost_column = "EUR"

            if month_column not in df.columns or cost_column not in df.columns:
                st.error(f"Columns '{month_column}' and '{cost_column}' not found! Please check column names.")
                st.stop()

            month_mapping = {
                "January": 1, "February": 2, "March": 3, "April": 4,
                "May": 5, "June": 6, "July": 7, "August": 8,
                "September": 9, "October": 10, "November": 11, "December": 12
            }

            df[month_column] = df[month_column].replace(month_mapping)
            df[month_column] = pd.to_numeric(df[month_column], errors="coerce")
            df[cost_column] = pd.to_numeric(df[cost_column], errors="coerce")

            df.dropna(subset=[month_column, cost_column], inplace=True)

            # Group by month and sum total cost per month
            monthly_data = df.groupby(month_column)[cost_column].sum().reset_index()
            monthly_data = monthly_data.sort_values(by=month_column)

            # ✅ Debugging Step: Show aggregated data
            st.write("### Aggregated Monthly Data (Total Cost Per Month)")
            st.dataframe(monthly_data)

            # Remove outliers: negative line item quantities
            monthly_data = monthly_data[monthly_data[cost_column] >= 0]

            # Extract numeric values
            months = monthly_data[month_column].values.astype(int)
            total_monthly_costs = monthly_data[cost_column].values.astype(float)

            # ✅ Ensure we have at least 12 months
            if len(total_monthly_costs) != 12:
                st.error(f"Data must contain exactly 12 months. Found: {len(total_monthly_costs)}.")
                st.stop()

            # Prepare LSTM training data
            X_train = np.array([total_monthly_costs])  # Use all 12 months as input
            y_train = np.array([total_monthly_costs[-1]])  # Predict the next month's cost

            # ✅ Debugging Step: Check shapes
            st.write(f"X_train shape: {X_train.shape}, y_train shape: {y_train.shape}")

            # Reshape input for LSTM
            X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)

            model = Sequential([
                LSTM(50, activation='relu', return_sequences=True, input_shape=(12, 1)),
                LSTM(25, activation='relu'),
                Dense(1)
            ])
            model.compile(optimizer='adam', loss='mse')
            model.fit(X_train, y_train, epochs=100, verbose=0)

            num_months = st.slider("Months to Predict", 1, 12, 12)

            # Predict future total monthly costs
            latest_data = total_monthly_costs.reshape(1, 12, 1)
            predicted_costs = []

            for i in range(num_months):
                prediction = model.predict(latest_data)

                if prediction.shape != (1, 1):
                    st.error(f"Unexpected prediction shape: {prediction.shape}. Stopping execution.")
                    st.stop()

                predicted_value = max(750000, float(prediction[0][0]))  # ✅ Clamp negative values
                predicted_costs.append(predicted_value)
                latest_data = np.roll(latest_data, -1, axis=1)
                latest_data[0, -1, 0] = prediction[0][0]

            # Generate future months extending from last historical month
            last_month = months[-1]
            future_months = [(last_month + i) % 12 or 12 for i in range(1, num_months + 1)]
            month_names = ["January", "February", "March", "April", "May", "June",
                           "July", "August", "September", "October", "November", "December"]
            future_labels = [month_names[m - 1] for m in future_months]

            future_month_numbers = np.arange(months[-1] + 1, months[-1] + num_months + 1)

            # Plot Predictions
            fig, ax = plt.subplots(figsize=(8, 4))
            ax.plot(months, total_monthly_costs, label="Historical Total Costs", marker='o', color='blue')
            ax.plot(future_month_numbers, predicted_costs, label="Predicted Total Costs", marker='o', linestyle="dashed", color='red')
            ax.set_title("Storage Cost Predictions")
            ax.set_xlabel("Month")
            ax.set_ylabel("Cost (€)")
            ax.set_xticks(np.concatenate([months, future_month_numbers]))
            ax.set_xticklabels(list(month_names[m - 1] for m in months) + future_labels, rotation=45)
            ax.legend()
            st.pyplot(fig)

            # Display Predictions in a Table
            st.write("### Predicted Storage Costs (€)")
            predicted_df = pd.DataFrame({"Month": future_labels, "Predicted Cost": predicted_costs})
            st.dataframe(predicted_df)

            # 🔹 Add manual feedback section
            st.write("### Provide Feedback on Prediction")
            feedback = st.selectbox(
                "Does the prediction seem accurate?",
                ["Accurate", "Too High", "Too Low"]
            )

            # 🔹 Rerun Button to adjust predictions
            if st.button("Rerun Model with Adjustment"):
                correction_factor = 1.0

                if feedback == "Too High":
                    correction_factor = 0.9  # Reduce predictions by 10%
                elif feedback == "Too Low":
                    correction_factor = 1.1  # Increase predictions by 10%

                predicted_costs = [cost * correction_factor for cost in predicted_costs]

                # 🔹 Display adjusted results
                fig, ax = plt.subplots(figsize=(8, 4))
                ax.plot(months, total_monthly_costs, label="Historical Total Costs", marker='o', color='blue')
                ax.plot(future_month_numbers, predicted_costs, label="Adjusted Predicted Costs", marker='o', linestyle="dashed", color='green')
                ax.set_title("Storage Cost Predictions (Adjusted)")
                ax.set_xlabel("Month")
                ax.set_ylabel("Cost (€)")
                ax.set_xticks(np.concatenate([months, future_month_numbers]))
                ax.set_xticklabels(list(month_names[m - 1] for m in months) + future_labels, rotation=45)
                ax.legend()
                st.pyplot(fig)

                st.write("### Adjusted Predicted Storage Costs (€)")
                adjusted_df = pd.DataFrame({"Month": future_labels, "Adjusted Cost": predicted_costs})
                st.dataframe(adjusted_df)

        except Exception as e:
            st.error(f"Error processing file: {e}")

