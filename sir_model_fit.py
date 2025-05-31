import pandas as pd
import numpy as np
from scipy.integrate import odeint
from scipy.optimize import least_squares
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import matplotlib.pyplot as plt

df = pd.read_csv("covid_19_clean_complete.csv")  
df["Date"] = pd.to_datetime(df["Date"])
global_df = df.groupby("Date")[["Confirmed"]].sum().reset_index()


t = np.arange(len(global_df)) 
N = 7_800_000_000  # 假設全球人口為 78 億

first_nonzero = global_df["Confirmed"].to_numpy().nonzero()[0][0]
I0 = global_df["Confirmed"].iloc[first_nonzero] / N 
S0 = 1.0 - I0 
R0 = 0.0  
y0 = [S0, I0, R0] 

# SIR 模型
def sir_model(y, t, beta, gamma):
    S, I, R = y
    dSdt = -beta * S * I
    dIdt = beta * S * I - gamma * I
    dRdt = gamma * I
    return [dSdt, dIdt, dRdt]

# 預測與誤差函數
def predict(params):
    beta, gamma = params
    result = odeint(sir_model, y0, t, args=(beta, gamma))
    S, I, R = result.T
    return N * (I + R)  #累積感染者數量

def residuals(params):
    return predict(params) - global_df["Confirmed"].values

# least_squares 擬合參數
initial_guess = [0.4, 0.1] 
bounds = ([0.0001, 0.0001], [np.inf, np.inf]) 
result = least_squares(residuals, initial_guess, bounds=bounds)
beta_fit, gamma_fit = result.x
predicted_cases = predict([beta_fit, gamma_fit])


r2 = r2_score(global_df["Confirmed"], predicted_cases)

print("最佳參數擬合結果：")
print(f"β（傳染率）   = {beta_fit:.4f}")
print(f"γ（康復率）   = {gamma_fit:.4f}")
print(f"R²             = {r2:.4f}")

plt.figure(figsize=(12, 6))
plt.plot(global_df["Date"], global_df["Confirmed"], label="Actual Confirmed", color="red")
plt.plot(global_df["Date"], predicted_cases, label="SIR Model Prediction", linestyle="--", color="blue")
plt.title("SIR Model Fit to Global COVID-19 Data")
plt.xlabel("Date")
plt.ylabel("Cumulative Confirmed Cases")
plt.xticks(rotation=45)
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("sir_model_fit.png", dpi=300)
print("圖片已儲存為 sir_model_fit.png")

S, I, R = odeint(sir_model, y0, t, args=(beta_fit, gamma_fit)).T

plt.figure(figsize=(12, 6))
plt.plot(global_df["Date"], S, label="S(t) Susceptible", color="green")
plt.plot(global_df["Date"], I, label="I(t) Infected", color="orange")
plt.plot(global_df["Date"], R, label="R(t) Recovered", color="blue")
plt.title("SIR Model Components Over Time (Proportions)")
plt.xlabel("Date")
plt.ylabel("Proportion of Population")
plt.ylim(0, 0.01) 
plt.xticks(rotation=45)
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("sir_model_components_proportion.png", dpi=300)
print("圖片已儲存為 sir_model_components_proportion.png")