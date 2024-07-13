import numpy as np
import pandas as pd
from statsmodels.tsa.holtwinters import SimpleExpSmoothing
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.metrics import r2_score
import matplotlib.pyplot as plt
import tkinter as tk
from tkinter import filedialog, messagebox

# Function to perform Simple Exponential Smoothing (SES)
def simple_exponential_smoothing(data, alpha):
    model = SimpleExpSmoothing(data)
    model_fit = model.fit(smoothing_level=alpha, optimized=False)
    smoothed_values = model_fit.fittedvalues
    return smoothed_values

# Function to forecast using SES
def forecast_with_ses(train_data, test_data, alpha):
    smoothed_values = simple_exponential_smoothing(train_data, alpha)
    forecast_horizon = len(test_data)
    forecast_values = [smoothed_values.iloc[-1]]  # Start forecast from the last smoothed value
    for i in range(forecast_horizon):
        next_forecast = alpha * test_data.iloc[i] + (1 - alpha) * forecast_values[-1]
        forecast_values.append(next_forecast)
    forecast_values = forecast_values[1:]  # Exclude the initial value used for starting forecast
    return forecast_values

# Function to calculate performance metrics
def calculate_metrics(actual_values, forecast_values):
    rmse = np.sqrt(mean_squared_error(actual_values, forecast_values))
    mae = mean_absolute_error(actual_values, forecast_values)
    mape = np.mean(np.abs((actual_values - forecast_values) / actual_values)) * 100
    r2 = r2_score(actual_values, forecast_values)
    return rmse, mae, mape, r2

# Function to plot actual vs. forecasted values
def plot_forecast(data, test_data, forecast_values):
    plt.figure(figsize=(10, 6))
    plt.plot(data.index, data["Vehicles"], label='Actual')
    plt.plot(test_data.index, forecast_values, label='Forecast', linestyle='--', color='red')
    plt.title("Simple Exponential Smoothing (SES) Forecast")
    plt.xlabel("Date")
    plt.ylabel("Vehicles")
    plt.legend()
    plt.show()

# Function to handle forecast and display results
def forecast_and_display(data, alpha):
    split_ratio = 0.8
    train_size = int(len(data) * split_ratio)
    train_data, test_data = data.iloc[:train_size]["Vehicles"], data.iloc[train_size:]["Vehicles"]

    forecast_values = forecast_with_ses(train_data, test_data, alpha)
    actual_values = test_data.values

    rmse, mae, mape, r2 = calculate_metrics(actual_values, forecast_values)

    # Display metrics in a message box
    message = f"RMSE: {rmse:.2f}\nMAE: {mae:.2f}\nMAPE: {mape:.2f}%\nR²: {r2:.2f}"
    messagebox.showinfo("Forecast Performance Metrics", message)

    # Plot forecasted values
    plot_forecast(data, test_data, forecast_values)

# Function to create a simple UI
def create_ui():
    root = tk.Tk()
    root.title("Simple Exponential Smoothing (SES) Forecasting")

    # Button to load data and perform forecasting
    def load_data_and_forecast():
        file_path = filedialog.askopenfilename(filetypes=[("CSV files", "*.csv")])
        if file_path:
            data = pd.read_csv(file_path)
            data["DateTime"] = pd.to_datetime(data["DateTime"], errors='coerce', format='%d-%m-%Y %H:%M')
            data.sort_values(by="DateTime", inplace=True)
            data.set_index("DateTime", inplace=True)

            alpha = float(alpha_entry.get())
            forecast_and_display(data, alpha)

    # Label and Entry for alpha value
    alpha_label = tk.Label(root, text="Enter alpha value (0 < alpha <= 1):")
    alpha_label.pack()
    alpha_entry = tk.Entry(root)
    alpha_entry.insert(0, "0.75")  # Default alpha value
    alpha_entry.pack()

    # Button to load data and perform forecasting
    forecast_button = tk.Button(root, text="Load Data and Forecast", command=load_data_and_forecast)
    forecast_button.pack()

    root.mainloop()

if __name__ == "__main__":
    # Call function to create the UI
    create_ui()
