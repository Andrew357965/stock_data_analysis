import yfinance as yf
import streamlit as st
from datetime import datetime
import pandas as pd
import numpy as np
import scipy.stats as stats
from scipy.stats import jarque_bera
from scipy.stats import skew, kurtosis
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from groq import Groq

# Fetch historical data for a given stock ticker
def get_stock_data(ticker, start_date, end_date):
    data = yf.download(ticker, start=start_date, end=end_date, auto_adjust=True)
    return data

# EDA
def eda(df):
    df = df.reset_index()
    df.columns = [col[0] if isinstance(col, tuple) else col for col in df.columns]
    df.rename(columns={
        'Date':   'date',
        'Close':  'close',
        'High':   'high',
        'Low':    'low',
        'Open':   'open',
        'Volume': 'volume'}, inplace=True)
    df['date']       = pd.to_datetime(df['date']).dt.tz_localize(None)
    df['range']      = df['high'] - df['low']
    df['price_diff'] = df['close'].diff()
    df['returns']    = df['close'].pct_change()
    df['volatility'] = df['returns'].rolling(10).std()
    df['volatility'] = df['volatility'].fillna(df['volatility'].mean())
    df = df.dropna().reset_index(drop=True)
    return df

def kalman_filter(z, Q, R):
    x_hat = [z[0]]
    P = [1.0]

    for t in range(1, len(z)):
        x_pred = x_hat[t-1]
        P_pred = P[t-1] + Q[t]

        K_k = P_pred / (P_pred + R[t])
        x_new = x_pred + K_k * (z[t] - x_pred)
        P_new = (1 - K_k) * P_pred

        x_hat.append(x_new)
        P.append(P_new)
    return P, x_hat

def kalman_smoother(P, x_hat, Q):
    T = len(x_hat)

    x_smooth = np.zeros(T)
    P_smooth = np.zeros(T)
    P_cross = np.zeros(T)
    x_smooth[-1] = x_hat[-1]
    P_smooth[-1] = P[-1]

    for t in range(T-2, -1, -1):
        P_pred = P[t] + Q[t]
        J_t = P[t] / P_pred
        x_smooth[t] = x_hat[t] + J_t*(x_smooth[t+1] - x_hat[t])
        P_smooth[t] = P[t] + (J_t**2)*(P_smooth[t+1] - P_pred)
        P_cross[t+1] = J_t * P_smooth[t+1]
    return x_smooth, P_smooth, P_cross

client = Groq(api_key=st.secrets["GROQ_API_KEY"])

st.title("Interactive Financial Stock Market Analysis Tool")
st.sidebar.header("User Input Options")
selected_stock = st.sidebar.text_input("Enter Stock Ticker", "AAPL").upper() # Default to Apple Inc.
start_date = st.sidebar.date_input("Enter Start Date", datetime(2022, 1, 1))
end_date = st.sidebar.date_input("Enter End Date", datetime(2024, 12, 31))

kf_params = st.sidebar.selectbox("Select parameters for Kalman Filter", ["Adaptive", "Constant (manually)"])

Q_value = st.sidebar.slider("Q", 0.0, 1.0, 0.01, 0.001, disabled=(kf_params == "Adaptive"))
R_value = st.sidebar.slider("R", 0.0, 10.0, 1.0, 0.1, disabled=(kf_params == "Adaptive"))
split_pct = st.sidebar.slider("Train/Test Split",
                               min_value=0.7,
                               max_value=0.9,
                               value=0.8,
                               step=0.05)

st.sidebar.markdown("---")
st.sidebar.markdown("**Author:** Andreas Katsaounis")

if start_date < end_date:
    stock_data = get_stock_data(selected_stock, start_date, end_date)

    if stock_data.empty:
        st.error("Invalid ticker or no data available.")
        st.stop()

    df = eda(stock_data)
    z = df['close'].values

    if kf_params == "Adaptive":
        R = (df['range']) ** 2
        alpha = 0.01
        Q = alpha * R
    else:
        Q = Q_value * np.ones(len(z))
        R = R_value * np.ones(len(z))

    P, x_hat = kalman_filter(z=z, Q=Q, R=R)

    residuals = np.array(z) - np.array(x_hat)

    x_hat = np.array(x_hat).flatten()
    x_smooth, P_smooth, P_cross = kalman_smoother(P=P, x_hat=x_hat, Q=Q)

    tab1, tab2, tab3, tab4 = st.tabs(["EDA", "Kalman Filter", "Classification models", "AI summary"])

    with tab1:
        st.subheader(f"{selected_stock} stock data from {start_date} to {end_date}")
        st.dataframe(df.head())
        st.write(f"Dataset shape: {df.shape}\n")
        st.subheader("Basic statistics")
        st.write(df.describe())

        fig, axes = plt.subplots(2, 2, figsize=(12, 8))

        axes[0, 0].plot(df['date'], z)
        axes[0, 0].set_title("Close price")
        axes[0, 0].tick_params(axis='x', rotation=45)

        axes[0, 1].plot(df['date'], df['returns'])
        axes[0, 1].tick_params(axis='x', rotation=45)
        axes[0, 1].set_title("Returns")

        sns.histplot(df['returns'], kde=True, ax=axes[1, 0])
        axes[1, 0].set_title("Histogram of returns")

        axes[1, 1].plot(df['date'], df['volatility'])
        axes[1, 1].tick_params(axis='x', rotation=45)
        axes[1, 1].set_title("Rolling Volatility")

        plt.tight_layout()
        st.pyplot(fig)
        plt.close(fig)

        if st.button("Generative AI Analysis"):
            close_min = float(df['close'].min().iloc[0]) if df['close'].ndim > 1 else float(df['close'].min())
            close_max = float(df['close'].max().iloc[0]) if df['close'].ndim > 1 else float(df['close'].max())
            prompt = f"""
            You are a financial data analyst. Analyze the following EDA results 
            for {selected_stock} stock and provide concise insights.
            BE STRICT ON OUTPUT LENGTH: max 150 words.

            Data:
            - Period: {start_date} to {end_date}
            - Mean return: {df['returns'].mean():.4f}
            - Return std: {df['returns'].std():.4f}
            - Mean volatility: {df['volatility'].mean():.4f}
            - Price range: {close_min:.2f} to {close_max:.2f}

            Comment on: trend, return distribution, volatility level.
            """
            try:
                response = client.chat.completions.create(
                    model="llama-3.3-70b-versatile",
                    messages=[{"role": "user", "content": prompt}]
                )
                st.write(response.choices[0].message.content)
            except Exception as e:
                st.error(f"API Error: {str(e)}")

    with tab2:
        st.header("Kalman Filter")


        st.write('''
        The Kalman Filter (KF) is a recursive algorithm that estimates the true 
        underlying state of a system from noisy observations. Applied to stock prices, 
        it separates the latent price signal from market noise.

        At each step, the filter balances two sources of information:
        - The model's prediction, governed by process noise Q
        - The new observation, governed by measurement noise R

        A smaller Q produces a smoother estimate; a smaller R trusts the observations 
        more. The Kalman Gain K_t determines this trade-off dynamically.

        In Adaptive mode, Q and R are time-varying, based on the daily price range 
        (high − low)², reflecting market uncertainty. The Kalman Smoother (KS), shown 
        at the end of this section, uses the full dataset to produce a lag-free 
        estimate — useful for visualisation but not for prediction.
        ''')

        residuals_mean = np.mean(residuals)
        residuals_std = np.std(residuals)
        error_magnitude = np.mean(residuals ** 2)
        root_mse = np.sqrt(error_magnitude)
        st.write(f"Error magnitude (MSE): {error_magnitude:.3f}")
        st.write(f"Sqrt of MSE: {root_mse:.3f}")

        fig, axes = plt.subplots(2, 1, sharex=True, figsize=(12, 10))
        axes[0].plot(df['date'], z, label="Observed raw data")
        axes[0].plot(df['date'], x_hat, label="Kalman (smoothed data)")
        axes[0].legend()
        axes[0].set_title("Kalman vs observed data")

        axes[1].plot(df['date'], residuals)
        axes[1].set_title("Residuals (noise)")

        for ax in axes:
            ax.tick_params(axis='x', rotation=45)

        st.pyplot(fig)
        plt.close(fig)

        x = np.linspace(min(residuals), max(residuals), 100)
        pdf = stats.norm.pdf(x, residuals_mean, residuals_std)

        fig, ax = plt.subplots(figsize=(12, 8))
        sns.histplot(residuals, bins=50, kde=True, stat="density", label="Residuals")
        ax.plot(x, pdf, color='red', label="Gaussian fit")
        ax.legend()
        ax.set_title("Residuals vs Gaussian fit")
        st.pyplot(fig)
        plt.close(fig)

        skew_val = skew(residuals).item()
        st.write(f"Skewness: {skew_val:.3f}")

        if abs(skew_val) < 0.5:
            st.write("Approximately symmetric")
        elif abs(skew_val) < 1:
            st.write("Moderate skewness")
        else:
            st.write("High skewness")

        st.markdown("---")
        kurt_val = kurtosis(residuals).item()
        st.write(f"Kurtosis: {kurt_val:.3f}")

        if abs(kurt_val) < 0.5:
            st.write("Close to Gaussian tails")
        elif kurt_val > 0:
            st.write("Fat tails (leptokurtic)")
        else:
            st.write("Thin tails (platykurtic)")

        jb_stat, p_value = jarque_bera(residuals)

        st.markdown("---")
        st.write("Jarque-Bera test")
        st.write(f"JB statistic: {jb_stat:.3f}")
        st.write(f"p-value: {p_value:.5f}")

        if p_value < 0.05:
            st.write("According to Jarque-Bera test, null hypothesis (H_0) is rejected;"
                     "data are not Gaussian.")
        else:
            st.write("According to Jarque-Bera test, null hypothesis (H_0) cannot be rejected.")

        with st.expander("Kalman Smoother"):
            st.subheader("Kalman Smoother")

            fig, ax = plt.subplots(figsize=(12, 4))
            ax.plot(df['date'], z, label="Observed raw data")
            ax.plot(df['date'], x_smooth, label="Kalman smoother")
            ax.tick_params(axis='x', rotation=45)
            ax.legend()
            ax.set_title("KS vs observed data")
            st.pyplot(fig)
            plt.close(fig)

        if st.button("AI Analysis: Kalman Filter"):
            prompt = f"""
            You are a financial data analyst. Analyze the following Kalman Filter results 
            for {selected_stock} stock. BE STRICT: max 150 words.

            Results:
            - MSE: {error_magnitude:.3f}
            - Sqrt MSE: {root_mse:.3f}
            - Residuals skewness: {skew_val:.3f}
            - Residuals kurtosis: {kurt_val:.3f}
            - Jarque-Bera p-value: {p_value:.5f}
            - KF parameters: {kf_params}

            Comment on: smoothing quality, Gaussian assumption validity, 
            what the residuals suggest about market noise.
            """
            try:
                response = client.chat.completions.create(
                    model="llama-3.3-70b-versatile",
                    messages=[{"role": "user", "content": prompt}]
                )
                st.write(response.choices[0].message.content)
            except Exception as e:
                st.error(f"API Error: {str(e)}")

    with tab3:
        st.header("Classification models")

        st.write('''
        Using features derived from the Kalman Filter (residuals, uncertainty) alongside 
        price-based features (lagged returns, volatility, intraday momentum), we train 
        three classification models to predict whether the next day's closing price will 
        increase or decrease.

        The naive baseline — always predicting "up" — sets the minimum bar to beat.

        - Logistic Regression: linear model, interpretable but limited to linear 
        decision boundaries.
        - Random Forest: ensemble of decision trees, robust to overfitting, provides 
        feature importance.
        - XGBoost: sequential boosting, corrects previous errors via gradient descent, 
        state-of-the-art for tabular data.
        ''')
        st.subheader("Logistic Regression")

        df1 = df.copy()
        df1['kf_close'] = x_hat
        df1['returns_lag1'] = df1['returns'].shift(1)
        df1['volatility_lag1'] = df1['volatility'].shift(1)
        df1['residuals'] = residuals
        df1['residuals_lag1'] = df1['residuals'].shift(1)
        df1['kf_uncertainty'] = P
        df1['kf_uncertainty_lag1'] = df1['kf_uncertainty'].shift(1)
        df1['direction'] = (df1['returns'] > 0).astype(int)
        df1['intraday_momentum'] = df1['close'] - df1['open']
        df1['volume_volatility'] = df1['volume'].pct_change().rolling(10).std()
        df1['volume_volatility'] = df1['volume_volatility'].fillna(df1['volume_volatility'].mean())
        df1['range_lag1'] = df1['range'].shift(1)
        df1['intraday_momentum_lag1'] = df1['intraday_momentum'].shift(1)
        df1['volume_volatility_lag1'] = df1['volume_volatility'].shift(1)
        df1['returns_rolling5'] = df1['returns'].rolling(5).mean().shift(1)
        df1['residuals_rolling5'] = df1['residuals'].rolling(5).mean().shift(1)
        df1['intraday_momentum5'] = df1['intraday_momentum'].rolling(5).mean().shift(1)
        df1['range_rolling5'] = df1['range'].rolling(5).mean().shift(1)
        df1 = df1.dropna().reset_index(drop=True)

        split_idx = int(split_pct * len(df1))

        train_df = df1.iloc[:split_idx]
        test_df = df1.iloc[split_idx:]

        features = ['returns_lag1', 'volatility_lag1', 'residuals_lag1',
                    'kf_uncertainty_lag1', 'range_lag1', 'intraday_momentum_lag1',
                    'volume_volatility_lag1', 'returns_rolling5', 'residuals_rolling5',
                    'intraday_momentum5', 'range_rolling5']
        X_train = train_df[features]
        y_train = train_df['direction']

        X_test = test_df[features]
        y_test = test_df['direction']

        scaler = StandardScaler()

        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        naive_acc = y_test.mean()
        st.write(f"Naive baseline accuracy: {naive_acc:.3f}")

        model = LogisticRegression()
        model.fit(X_train_scaled, y_train)

        y_pred_lr = model.predict(X_test_scaled)

        st.write(f"Accuracy: {accuracy_score(y_test, y_pred_lr):.3f}")

        fig, ax = plt.subplots(figsize=(8, 6))
        cm = confusion_matrix(y_test, y_pred_lr)
        ConfusionMatrixDisplay(cm).plot(ax=ax)
        plt.title("Logistic Regression")
        st.pyplot(fig)
        plt.close(fig)

        st.subheader("Random Forest")
        model = RandomForestClassifier(n_estimators=100, random_state=42)
        model.fit(X_train_scaled, y_train)
        y_pred_rf = model.predict(X_test_scaled)

        st.write(f"Accuracy: {accuracy_score(y_test, y_pred_rf):.3f}")

        fig, ax = plt.subplots(figsize=(12, 8))
        importances = pd.Series(model.feature_importances_, index=features)
        importances.sort_values().plot(kind='barh', ax=ax)
        ax.set_title("Random Forest - Feature Importance")
        st.pyplot(fig)
        plt.close(fig)

        st.subheader("XGBoost")
        xgb = XGBClassifier(n_estimators=100, random_state=42, eval_metric='logloss')
        xgb.fit(X_train_scaled, y_train)
        y_pred_xgb = xgb.predict(X_test_scaled)

        st.write(f"XGB Accuracy: {accuracy_score(y_test, y_pred_xgb):.3f}")

        st.subheader("Model Comparison")
        results = pd.DataFrame({
            'Model': ['Naive Baseline', 'Logistic Regression', 'Random Forest', 'XGBoost'],
            'Accuracy': [
                round(naive_acc, 3),
                round(accuracy_score(y_test, y_pred_lr), 3),
                round(accuracy_score(y_test, y_pred_rf), 3),
                round(accuracy_score(y_test, y_pred_xgb), 3)
            ]
        })
        st.write(results)

        plt.figure(figsize=(12,8))
        results.set_index('Model')['Accuracy'].plot(kind='bar', ylim=(0.1, 0.7))
        plt.axhline(naive_acc, color='red', linestyle='--', label='Baseline')
        plt.title("Model Comparison")
        plt.ylabel("Accuracy")
        plt.xticks(rotation=30)
        plt.legend()
        st.pyplot(plt.gcf())
        plt.close(fig)

        if st.button("AI Analysis: Classification Models"):
            prompt = f"""
            You are a financial data analyst. Analyze the following ML classification 
            results for {selected_stock} stock. BE STRICT: max 150 words.

            Results:
            - Naive baseline accuracy: {naive_acc:.3f}
            - Logistic Regression accuracy: {accuracy_score(y_test, y_pred_lr):.3f}
            - Random Forest accuracy: {accuracy_score(y_test, y_pred_rf):.3f}
            - XGBoost accuracy: {accuracy_score(y_test, y_pred_xgb):.3f}
            - Train/test split: {split_pct:.0%} / {1 - split_pct:.0%}

            Comment on: whether any model beats the baseline, what this implies 
            about market predictability, connection to Efficient Market Hypothesis.
            """
            try:
                response = client.chat.completions.create(
                    model="llama-3.3-70b-versatile",
                    messages=[{"role": "user", "content": prompt}]
                )
                st.write(response.choices[0].message.content)
            except Exception as e:
                st.error(f"API Error: {str(e)}")
    with tab4:
        st.header("AI Summary")
        st.write("Click below for a complete analysis of all results.")

        if st.button("Generate Full AI Summary"):
            prompt = f"""
            You are a financial data analyst. Write a complete analysis report 
            for {selected_stock} from {start_date} to {end_date}. 
            BE STRICT: max 300 words. Structure: EDA, Kalman Filter, ML Models, Conclusion.

            Data:
            - Mean return: {df['returns'].mean():.4f}
            - Return std: {df['returns'].std():.4f}
            - Mean volatility: {df['volatility'].mean():.4f}
            - KF MSE: {error_magnitude:.3f}
            - Residuals skewness: {skew_val:.3f}
            - Residuals kurtosis: {kurt_val:.3f}
            - JB p-value: {p_value:.5f}
            - Naive baseline: {naive_acc:.3f}
            - LR accuracy: {accuracy_score(y_test, y_pred_lr):.3f}
            - RF accuracy: {accuracy_score(y_test, y_pred_rf):.3f}
            - XGBoost accuracy: {accuracy_score(y_test, y_pred_xgb):.3f}

            Connect findings to Efficient Market Hypothesis.
            """
            try:
                response = client.chat.completions.create(
                    model="llama-3.3-70b-versatile",
                    messages=[{"role": "user", "content": prompt}]
                )
                st.write(response.choices[0].message.content)
            except Exception as e:
                st.error(f"API Error: {str(e)}")
else:
    st.error("Error: End date must fall after start date.")
    st.stop()
