
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)

# PAGE CONFIGURATION

st.set_page_config(
    page_title="Boston House Price Prediction",
    page_icon="🏠",
    layout="wide"
)

# TITLE

st.title("🏠 Boston House Price Prediction")

st.subheader(
    "Multiple Linear Regression using Boston Housing Dataset"
)

st.write(
    "This application predicts MEDV using multiple "
    "housing-related attributes."
)

st.divider()



# LOAD EXCEL DATASET


@st.cache_data
def load_data():

    df = pd.read_excel(
        "boston_housing_clean.xlsx"
    )

    # Remove unwanted unnamed columns
    df = df.loc[
        :,
        ~df.columns.astype(str).str.startswith("Unnamed")
    ]

    return df


try:

    df = load_data()

except Exception as e:

    st.error(
        "Unable to load the Excel file."
    )

    st.write(
        "Make sure Boston_Housing.xlsx is in the same "
        "folder as app.py."
    )

    st.exception(e)

    st.stop()


# FIND MEDV COLUMN


target_column = None

for column in df.columns:

    if str(column).strip().upper() == "MEDV":

        target_column = column
        break


if target_column is None:

    st.error("MEDV column was not found in the Excel file.")

    st.write("Available columns:")

    st.write(
        df.columns.tolist()
    )

    st.stop()


# SIDEBAR


st.sidebar.title("Navigation")

page = st.sidebar.radio(
    "Go to",
    [
        "Dataset Overview",
        "Exploratory Data Analysis",
        "Model Development",
        "Model Evaluation",
        "Prediction"
    ]
)

# ============================================================
# DATASET OVERVIEW
# ============================================================

if page == "Dataset Overview":

    st.header("📊 Dataset Overview")

    # Dataset dimensions

    col1, col2 = st.columns(2)

    with col1:

        st.metric(
            "Number of Rows",
            df.shape[0]
        )

    with col2:

        st.metric(
            "Number of Columns",
            df.shape[1]
        )


    st.subheader("First 5 Records")

    st.dataframe(
        df.head(),
        use_container_width=True
    )


    # Attribute names

    st.subheader("Attribute Names")

    st.write(
        df.columns.tolist()
    )


    # Data types

    st.subheader("Attribute Data Types")

    datatype_df = pd.DataFrame({

        "Attribute": df.columns,

        "Data Type": [
            str(dtype)
            for dtype in df.dtypes
        ]

    })

    st.dataframe(
        datatype_df,
        use_container_width=True
    )


    # Missing values

    st.subheader("Missing Values")

    missing_df = pd.DataFrame({

        "Attribute": df.columns,

        "Missing Values": df.isnull().sum().values

    })

    st.dataframe(
        missing_df,
        use_container_width=True
    )


    if df.isnull().sum().sum() == 0:

        st.success(
            "No missing values found."
        )

    else:

        st.warning(
            "Missing values are present."
        )


    # Duplicate records

    st.subheader("Duplicate Records")

    duplicate_count = df.duplicated().sum()

    st.write(
        "Number of duplicate records:",
        duplicate_count
    )


    if duplicate_count == 0:

        st.success(
            "No duplicate records found."
        )

    else:

        st.warning(
            "Duplicate records are present."
        )


    # Statistical information

    st.subheader(
        "Basic Statistical Information"
    )

    st.dataframe(
        df.describe(),
        use_container_width=True
    )



# EXPLORATORY DATA ANALYSIS


elif page == "Exploratory Data Analysis":

    st.header(
        "📈 Exploratory Data Analysis"
    )



    # MEDV DISTRIBUTION
    

    st.subheader(
        "Distribution of MEDV"
    )

    fig, ax = plt.subplots(
        figsize=(9, 5)
    )

    sns.histplot(
        df[target_column],
        kde=True,
        ax=ax
    )

    ax.set_xlabel(
        "MEDV"
    )

    ax.set_ylabel(
        "Frequency"
    )

    ax.set_title(
        "Distribution of MEDV"
    )

    st.pyplot(fig)



    # CORRELATION
   

    st.subheader(
        "Correlation Matrix"
    )

    correlation = df.corr(
        numeric_only=True
    )

    fig, ax = plt.subplots(
        figsize=(12, 9)
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



    # CORRELATION WITH MEDV


    st.subheader(
        "Correlation of Attributes with MEDV"
    )

    medv_corr = (
        correlation[target_column]
        .sort_values(
            ascending=False
        )
    )

    st.dataframe(
        medv_corr.to_frame(
            name="Correlation with MEDV"
        ),
        use_container_width=True
    )


    # SELECT FEATURE


    feature_columns = [
        column
        for column in df.columns
        if column != target_column
        and pd.api.types.is_numeric_dtype(
            df[column]
        )
    ]


    selected_feature = st.selectbox(
        "Select an attribute to visualize its relationship with MEDV:",
        feature_columns
    )


    fig, ax = plt.subplots(
        figsize=(9, 5)
    )

    sns.scatterplot(
        data=df,
        x=selected_feature,
        y=target_column,
        ax=ax
    )

    ax.set_title(
        selected_feature
        + " vs MEDV"
    )

    st.pyplot(fig)


# MODEL DEVELOPMENT


elif page == "Model Development":

    st.header(
        "🤖 Multiple Linear Regression Model"
    )


    # Separate X and y

    X = df.drop(
        target_column,
        axis=1
    )

    y = df[target_column]


    # Keep only numeric predictors

    X = X.select_dtypes(
        include=np.number
    )


    # Fill missing values

    X = X.fillna(
        X.median()
    )


    # Train-test split

    X_train, X_test, y_train, y_test = train_test_split(

        X,
        y,

        test_size=0.20,

        random_state=42
    )


    # Model

    model = LinearRegression()

    model.fit(
        X_train,
        y_train
    )


    # Predictions

    y_train_pred = model.predict(
        X_train
    )

    y_test_pred = model.predict(
        X_test
    )


    st.success(
        "Multiple Linear Regression model trained successfully."
    )


    # Dataset split

    col1, col2 = st.columns(2)

    with col1:

        st.metric(
            "Training Records",
            X_train.shape[0]
        )

    with col2:

        st.metric(
            "Testing Records",
            X_test.shape[0]
        )


    # Intercept

    st.subheader(
        "Model Intercept"
    )

    st.write(
        model.intercept_
    )


    # Coefficients

    st.subheader(
        "Regression Coefficients"
    )

    coefficient_df = pd.DataFrame({

        "Feature": X.columns,

        "Coefficient": model.coef_

    })


    coefficient_df[
        "Absolute Coefficient"
    ] = coefficient_df[
        "Coefficient"
    ].abs()


    coefficient_df = coefficient_df.sort_values(
        by="Absolute Coefficient",
        ascending=False
    )


    st.dataframe(
        coefficient_df,
        use_container_width=True
    )


    st.info(
        "Positive coefficients indicate a positive relationship "
        "with MEDV, while negative coefficients indicate a "
        "negative relationship, keeping other variables constant."
    )

# MODEL EVALUATION

elif page == "Model Evaluation":

    st.header(
        "📊 Model Evaluation"
    )


    # Prepare data

    X = df.drop(
        target_column,
        axis=1
    )

    y = df[target_column]


    X = X.select_dtypes(
        include=np.number
    )

    X = X.fillna(
        X.median()
    )


    # Split

    X_train, X_test, y_train, y_test = train_test_split(

        X,
        y,

        test_size=0.20,

        random_state=42
    )


    # Train

    model = LinearRegression()

    model.fit(
        X_train,
        y_train
    )


    # Predictions

    y_train_pred = model.predict(
        X_train
    )

    y_test_pred = model.predict(
        X_test
    )


    # Metrics

    mae = mean_absolute_error(
        y_test,
        y_test_pred
    )

    mse = mean_squared_error(
        y_test,
        y_test_pred
    )

    rmse = np.sqrt(
        mse
    )

    r2 = r2_score(
        y_test,
        y_test_pred
    )


    # Adjusted R2

    n = X_test.shape[0]

    p = X_test.shape[1]

    adjusted_r2 = 1 - (

        ((1 - r2) * (n - 1))

        /

        (n - p - 1)

    )


    # DISPLAY METRICS
 

    st.subheader(
        "Regression Metrics"
    )

    col1, col2, col3, col4, col5 = st.columns(5)

    with col1:

        st.metric(
            "MAE",
            f"{mae:.4f}"
        )

    with col2:

        st.metric(
            "MSE",
            f"{mse:.4f}"
        )

    with col3:

        st.metric(
            "RMSE",
            f"{rmse:.4f}"
        )

    with col4:

        st.metric(
            "R² Score",
            f"{r2:.4f}"
        )

    with col5:

        st.metric(
            "Adjusted R²",
            f"{adjusted_r2:.4f}"
        )



    # TRAIN VS TEST


    st.subheader(
        "Training vs Testing Performance"
    )

    train_r2 = r2_score(
        y_train,
        y_train_pred
    )

    test_r2 = r2_score(
        y_test,
        y_test_pred
    )


    train_rmse = np.sqrt(
        mean_squared_error(
            y_train,
            y_train_pred
        )
    )

    test_rmse = np.sqrt(
        mean_squared_error(
            y_test,
            y_test_pred
        )
    )


    performance_df = pd.DataFrame({

        "Dataset": [
            "Training",
            "Testing"
        ],

        "R² Score": [
            train_r2,
            test_r2
        ],

        "RMSE": [
            train_rmse,
            test_rmse
        ]

    })


    st.dataframe(
        performance_df,
        use_container_width=True
    )


    difference = train_r2 - test_r2


    if difference > 0.10:

        st.warning(
            "The difference between training and testing "
            "R² suggests possible overfitting."
        )

    elif train_r2 < 0.50 and test_r2 < 0.50:

        st.warning(
            "Both training and testing R² are relatively low, "
            "which may indicate underfitting."
        )

    else:

        st.success(
            "Training and testing performance are reasonably close."
        )


    # ACTUAL VS PREDICTED


    st.subheader(
        "Actual vs Predicted MEDV"
    )

    fig, ax = plt.subplots(
        figsize=(9, 6)
    )

    ax.scatter(
        y_test,
        y_test_pred,
        alpha=0.7
    )


    minimum = min(
        y_test.min(),
        y_test_pred.min()
    )

    maximum = max(
        y_test.max(),
        y_test_pred.max()
    )


    ax.plot(
        [minimum, maximum],
        [minimum, maximum]
    )


    ax.set_xlabel(
        "Actual MEDV"
    )

    ax.set_ylabel(
        "Predicted MEDV"
    )

    ax.set_title(
        "Actual vs Predicted MEDV"
    )

    st.pyplot(fig)


    # RESIDUAL PLOT


    st.subheader(
        "Residual Plot"
    )

    residuals = (
        y_test - y_test_pred
    )


    fig, ax = plt.subplots(
        figsize=(9, 6)
    )

    ax.scatter(
        y_test_pred,
        residuals,
        alpha=0.7
    )

    ax.axhline(
        y=0
    )

    ax.set_xlabel(
        "Predicted MEDV"
    )

    ax.set_ylabel(
        "Residual"
    )

    ax.set_title(
        "Residual Plot"
    )

    st.pyplot(fig)


    # RESIDUAL DISTRIBUTION
    

    st.subheader(
        "Residual Distribution"
    )

    fig, ax = plt.subplots(
        figsize=(9, 5)
    )

    sns.histplot(
        residuals,
        kde=True,
        ax=ax
    )

    ax.set_xlabel(
        "Residual"
    )

    ax.set_ylabel(
        "Frequency"
    )

    ax.set_title(
        "Distribution of Residuals"
    )

    st.pyplot(fig)



# NEW HOUSE PREDICTION


elif page == "Prediction":

    st.header(
        "🏡 Predict House Price"
    )


    # Prepare data

    X = df.drop(
        target_column,
        axis=1
    )

    y = df[target_column]


    X = X.select_dtypes(
        include=np.number
    )

    X = X.fillna(
        X.median()
    )


    # Train model

    X_train, X_test, y_train, y_test = train_test_split(

        X,
        y,

        test_size=0.20,

        random_state=42
    )


    model = LinearRegression()

    model.fit(
        X_train,
        y_train
    )


    st.write(
        "Enter values for the house attributes below."
    )


    # Input fields

    input_values = {}


    for feature in X.columns:

        default_value = float(
            X[feature].mean()
        )


        input_values[feature] = st.number_input(

            feature,

            value=default_value

        )


    # Prediction button

    if st.button(
        "🔮 Predict MEDV"
    ):

        new_house = pd.DataFrame(
            [input_values]
        )


        prediction = model.predict(
            new_house
        )


        st.success(
            f"Predicted MEDV: {prediction[0]:.2f}"
        )


        st.info(
            "MEDV represents the median value of "
            "owner-occupied homes in the original dataset."
        )




