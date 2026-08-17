import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

st.set_page_config(
    page_title="Automated EDA Tool",
    page_icon="📊",
    layout="wide"
)

#Title to be displayed on the webpage
st.title("📊 Automated Exploratory Data Analysis Tool")
st.markdown(
    """
    Upload a **CSV or Excel file** and this tool will automatically
    perform data inspection, cleaning, statistical analysis,
    visualization and basic insight generation.
    """
)

#File uploader to accept a file
uploaded_file = st.file_uploader(
    "Upload your dataset",
    type=["csv", "xls", "xlsx"]
)

#Main program of EDA
if uploaded_file is not None:

    try:
#reading the uploaded file 
        file_name = uploaded_file.name.lower()

        if file_name.endswith(".csv"):
            df = pd.read_csv(uploaded_file)

        elif file_name.endswith(".xls"):
            df = pd.read_excel(uploaded_file, engine="xlrd")

        elif file_name.endswith(".xlsx"):
            df = pd.read_excel(uploaded_file, engine="openpyxl")

        else:
            st.error("Unsupported file format.")
            st.stop()

    except Exception as e:

        st.error(f"Error while loading the file: {e}")
        st.stop()

#Dataset overview
    st.header("Dataset Overview")
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric("Rows", df.shape[0])

    with col2:
        st.metric("Columns", df.shape[1])

    with col3:
        st.metric(
            "Missing Values",
            int(df.isnull().sum().sum())
        )

    with col4:
        st.metric(
            "Duplicate Rows",
            int(df.duplicated().sum())
        )


#Dataset preview
    st.subheader("Dataset Preview")

    st.dataframe(
        df.head(10),
        use_container_width=True
    )

    st.subheader("Column Information")

    column_info = pd.DataFrame({
        "Column": df.columns,
        "Data Type": df.dtypes.astype(str),
        "Missing Values": df.isnull().sum().values,
        "Unique Values": [
            df[col].nunique()
            for col in df.columns
        ]
    })

    st.dataframe(
        column_info,
        use_container_width=True
    )

#Data cleaning
    st.header(" Data Cleaning")
    clean_df = df.copy()
    missing_before = clean_df.isnull().sum().sum()
    duplicate_before = clean_df.duplicated().sum()


#Remove duplicate rows
    clean_df = clean_df.drop_duplicates()

#Fill missing numerical values with median
    numerical_columns = clean_df.select_dtypes(
        include=np.number
    ).columns
    for column in numerical_columns:
        if clean_df[column].isnull().sum() > 0:
            clean_df[column] = clean_df[column].fillna(
                clean_df[column].median()
            )


#Fill missing categorical values with mode
    categorical_columns = clean_df.select_dtypes(
        include=["object", "category"]
    ).columns
    for column in categorical_columns:
        if clean_df[column].isnull().sum() > 0:
            mode_value = clean_df[column].mode()
            if len(mode_value) > 0:
                clean_df[column] = clean_df[column].fillna(
                    mode_value[0]
                )

    missing_after = clean_df.isnull().sum().sum()
    duplicate_after = clean_df.duplicated().sum()

#After cleaning summary
    clean_col1, clean_col2, clean_col3, clean_col4 = st.columns(4)

    with clean_col1:
        st.metric(
            "Missing Before",
            int(missing_before)
        )

    with clean_col2:

        st.metric(
            "Missing After",
            int(missing_after)
        )

    with clean_col3:

        st.metric(
            "Duplicates Before",
            int(duplicate_before)
        )

    with clean_col4:

        st.metric(
            "Duplicates After",
            int(duplicate_after)
        )

#Complete summary
    st.header("Statistical Summary")

    st.dataframe(
        clean_df.describe(include="all").T,
        use_container_width=True
    )

#Numeric Analysis
    st.header("Numerical Analysis")

    numerical_columns = clean_df.select_dtypes(
        include=np.number
    ).columns.tolist()


    if len(numerical_columns) > 0:
        selected_numeric = st.selectbox(
            "Select a numerical column",
            numerical_columns
        )


# Histogram
        st.subheader(
            f"Distribution of {selected_numeric}"
        )
        fig, ax = plt.subplots(figsize=(10, 5))
        sns.histplot(
            clean_df[selected_numeric],
            kde=True,
            ax=ax
        )
        ax.set_xlabel(selected_numeric)
        ax.set_ylabel("Frequency")

        st.pyplot(fig)
        plt.close(fig)


        # Box plot
        st.subheader(
            f"Box Plot of {selected_numeric}"
        )
        fig, ax = plt.subplots(figsize=(10, 4))
        sns.boxplot(
            x=clean_df[selected_numeric],
            ax=ax
        )

        st.pyplot(fig)
        plt.close(fig)


    else:

        st.warning(
            "No numerical columns found in the provided data."
        )

#category analysis
    st.header("Categorical Analysis")

    categorical_columns = clean_df.select_dtypes(
        include=["object", "category"]
    ).columns.tolist()

    if len(categorical_columns) > 0:

        selected_category = st.selectbox(
            "Select a categorical column",
            categorical_columns
        )

        value_counts = (
            clean_df[selected_category]
            .value_counts()
            .head(15)
        )

        fig, ax = plt.subplots(
            figsize=(10, 5)
        )

        sns.barplot(
            x=value_counts.values,
            y=value_counts.index,
            ax=ax
        )

        ax.set_xlabel("Count")
        ax.set_ylabel(selected_category)
        ax.set_title(
            f"Top Categories in {selected_category}"
        )

        st.pyplot(fig)
        plt.close(fig)

    else:

        st.warning(
            "No categorical columns found."
        )

#Correlation analysis
    st.header("Correlation Analysis")
    if len(numerical_columns) >= 2:

        correlation = clean_df[
            numerical_columns
        ].corr()

        fig, ax = plt.subplots(
            figsize=(10, 7)
        )

        sns.heatmap(
            correlation,
            annot=True,
            cmap="coolwarm",
            fmt=".2f",
            ax=ax
        )

        ax.set_title(
            "Correlation Heatmap"
        )

        st.pyplot(fig)
        plt.close(fig)

    else:

        st.warning(
            "At least two numerical columns are required for correlation."
        )


#Outlier detection
    st.header("Outlier Detection")
    if len(numerical_columns) > 0:
        outlier_summary = []

        for column in numerical_columns:

            Q1 = clean_df[column].quantile(0.25)
            Q3 = clean_df[column].quantile(0.75)
            IQR = Q3 - Q1
            lower_limit = Q1 - 1.5 * IQR
            upper_limit = Q3 + 1.5 * IQR
            outliers = clean_df[
                (clean_df[column] < lower_limit)
                |
                (clean_df[column] > upper_limit)
            ]

            outlier_summary.append({
                "Column": column,
                "Outliers": len(outliers),
                "Percentage": round(
                    len(outliers)
                    / len(clean_df)
                    * 100,
                    2
                )
            })


        outlier_df = pd.DataFrame(
            outlier_summary
        )

        st.dataframe(
            outlier_df,
            use_container_width=True
        )

#Automatic insights
    st.header("Automatic Insights")
    insights = []

    insights.append(
        f"The dataset contains "
        f"**{clean_df.shape[0]:,} rows** and "
        f"**{clean_df.shape[1]} columns**."
    )

    if missing_before > 0:

        insights.append(
            f"The dataset initially contained "
            f"**{missing_before:,} missing values**, "
            f"which were handled during preprocessing."
        )

    else:

        insights.append(
            "The dataset contains no missing values."
        )

    # Duplicate values
    if duplicate_before > 0:

        insights.append(
            f"**{duplicate_before:,} duplicate rows** "
            f"were detected and removed."
        )

    else:

        insights.append(
            "No duplicate rows were detected."
        )

    # Numerical insights
    for column in numerical_columns[:5]:

        mean_value = clean_df[column].mean()

        max_value = clean_df[column].max()

        min_value = clean_df[column].min()


        insights.append(
            f"For **{column}**, the average is "
            f"**{mean_value:.2f}**, with a minimum of "
            f"**{min_value:.2f}** and maximum of "
            f"**{max_value:.2f}**."
        )

    # Display insights
    for insight in insights:

        st.markdown(
            f"- {insight}"
        )

#To download the clean data
    st.header("⬇ Download Clean Dataset")

    csv_data = clean_df.to_csv(
        index=False
    ).encode("utf-8")

    st.download_button(
        label="Download Clean CSV",
        data=csv_data,
        file_name="cleaned_dataset.csv",
        mime="text/csv"
    )


    st.markdown("---")

    st.markdown(" Automated EDA Tool /n  The tool automatically performs data inspection,preprocessing, statistical analysis, visualization,outlier detection and basic insight generation.")