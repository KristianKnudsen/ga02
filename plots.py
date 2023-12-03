import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.optimize import curve_fit

# Read the CSV files
df_base = pd.read_csv('model_logs/PYV21.csv')
df_new = pd.read_csv('model_logs/TFv18.csv')

# Calculate the moving averages
df_base['length_mean_ma'] = df_base['length_mean'].rolling(10).mean()
df_new['length_mean_ma'] = df_new['length_mean'].rolling(10).mean()

# Adjusted sigmoid function definition
def sigmoid(x, L, x0, k):
    b = 2 - L / (1 + np.exp(k * x0))  # Adjust b based on L, x0, and k
    y = L / (1 + np.exp(-k * (x - x0))) + b
    return y

# Function to perform curve fitting
def fit_sigmoid(df):
    df_filtered = df.dropna(subset=['length_mean_ma'])
    X = df_filtered['iteration']
    y = df_filtered['length_mean_ma']
    L_guess = max(y) - 2  # Adjusted guess for L
    x0_guess = 0          # Since we want the curve to cross y=2 at x=0
    k_guess = 0.000001
    p0 = [L_guess, x0_guess, k_guess]
    popt, _ = curve_fit(sigmoid, X, y, p0=p0, maxfev=10000)
    return popt, df_filtered

# Fit sigmoid to both datasets
popt_base, df_filtered_base = fit_sigmoid(df_base)
popt_new, df_filtered_new = fit_sigmoid(df_new)

# Use fitted parameters to predict values
df_filtered_base['length_mean_ma_predicted'] = sigmoid(df_filtered_base['iteration'], *popt_base)
df_filtered_new['length_mean_ma_predicted'] = sigmoid(df_filtered_new['iteration'], *popt_new)


# Create a plot
fig, axs = plt.subplots(figsize=(8, 8))
axs.set_title('Snake Mean Length Comparison')

# Plot the raw data first with different styles and light colors
axs.plot(df_base['iteration'], df_base['length_mean'], label='BS64 PyTorch v21 Raw', color='gray', alpha=0.5, linestyle='--', zorder=1)
axs.plot(df_new['iteration'], df_new['length_mean'], label='BS64 TF v18 Raw', color='silver', alpha=0.5, linestyle=':', zorder=1)

# Plot the moving averages with similar but weaker and darker colors
axs.plot(df_base['iteration'], df_base['length_mean_ma'], label='PyTorch v21 Moving Average', color='midnightblue', alpha=0.7, linewidth=1.5, zorder=2)
axs.plot(df_new['iteration'], df_new['length_mean_ma'], label='TF v18 Moving Average', color='maroon', alpha=0.7, linewidth=1.5, zorder=2)

# Plot the sigmoid regressions on top with bright colors
axs.plot(df_filtered_base['iteration'], df_filtered_base['length_mean_ma_predicted'], label='Sigmoid Regression v21', color='blue', linewidth=2, zorder=3)
axs.plot(df_filtered_new['iteration'], df_filtered_new['length_mean_ma_predicted'], label='Sigmoid Regression v18', color='red', linewidth=2, zorder=3)

# Set the labels for the axes
axs.set_ylabel('Mean Length')
axs.set_xlabel('Iteration')

# Calculate convergence values and round to one decimal place
convergence_value_base = popt_base[0] + (2 - popt_base[0] / (1 + np.exp(popt_base[2] * popt_base[1])))
convergence_value_new = popt_new[0] + (2 - popt_new[0] / (1 + np.exp(popt_new[2] * popt_new[1])))

# Add convergence values to the plot
# Positioning further down in the bottom right
text_position_x = max(max(df_base['iteration']), max(df_new['iteration'])) * 0.8
text_position_y_base = min(convergence_value_base, convergence_value_new) - 10  # Lower position
text_position_y_new = text_position_y_base - 3  # Slightly lower than the previous line

axs.text(text_position_x, text_position_y_base, f"Convergence (PyTorch v21): {convergence_value_base:.1f}", color='blue', fontsize=10)
axs.text(text_position_x, text_position_y_new, f"Convergence (TF v18): {convergence_value_new:.1f}", color='red', fontsize=10)


# Show the legend
plt.legend()

# Display the plot
plt.show()