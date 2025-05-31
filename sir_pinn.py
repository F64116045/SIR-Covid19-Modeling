import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import pandas as pd
from scipy.interpolate import interp1d
from google.colab import files

uploaded = files.upload()

df = pd.read_csv("covid_19_clean_complete.csv")
df["Date"] = pd.to_datetime(df["Date"])
global_data = df.groupby("Date")["Confirmed"].sum().reset_index()

confirmed = global_data["Confirmed"].values.astype(np.float32)
t_data = np.arange(len(confirmed), dtype=np.float32).reshape(-1, 1)
confirmed_norm = confirmed / confirmed.max()
t_norm = t_data / t_data.max()

confirmed_interp = interp1d(t_norm.flatten(), confirmed_norm, fill_value="extrapolate")

class SIR_PINN(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(1, 64), nn.Tanh(),
            nn.Linear(64, 64), nn.Tanh(),
            nn.Linear(64, 3),
            nn.Sigmoid()
        )

    def forward(self, t):
        return self.net(t)


raw_beta = torch.tensor(0.5, requires_grad=True)
raw_gamma = torch.tensor(1/3, requires_grad=True)

def get_beta_gamma():
    return torch.nn.functional.softplus(raw_beta), torch.nn.functional.softplus(raw_gamma)


def pinn_loss(model, t_tensor, beta, gamma):
    t_tensor.requires_grad_(True)
    sir = model(t_tensor)
    S, I, R = sir[:, 0:1], sir[:, 1:2], sir[:, 2:3]

    dS = torch.autograd.grad(S, t_tensor, grad_outputs=torch.ones_like(S), create_graph=True)[0]
    dI = torch.autograd.grad(I, t_tensor, grad_outputs=torch.ones_like(I), create_graph=True)[0]
    dR = torch.autograd.grad(R, t_tensor, grad_outputs=torch.ones_like(R), create_graph=True)[0]

    ode1 = dS + beta * S * I
    ode2 = dI - beta * S * I + gamma * I
    ode3 = dR - gamma * I

    with torch.no_grad():
        i_r_true = torch.tensor(confirmed_interp(t_tensor.detach().cpu().numpy().flatten()), dtype=torch.float32).reshape(-1, 1)
    i_r_pred = I + R

    loss = (
        torch.mean(ode1 ** 2) +
        torch.mean(ode2 ** 2) +
        torch.mean(ode3 ** 2) +
        torch.mean((i_r_pred - i_r_true) ** 2)
    )
    return loss


model = SIR_PINN()
t_tensor = torch.tensor(t_norm, dtype=torch.float32)
optimizer = torch.optim.Adam(list(model.parameters()) + [raw_beta, raw_gamma], lr=1e-3)


loss_history = []
for epoch in range(3000):
    optimizer.zero_grad()
    beta, gamma = get_beta_gamma()
    loss = pinn_loss(model, t_tensor, beta, gamma)
    loss.backward()
    optimizer.step()
    loss_history.append(loss.item())

    if epoch % 300 == 0:
        print(f"Epoch {epoch}: Loss = {loss.item():.6f}, beta = {beta.item():.4f}, gamma = {gamma.item():.4f}")


t_plot = torch.linspace(0, 1, 200).reshape(-1, 1)
sir_pred = model(t_plot).detach().numpy()


plt.figure(figsize=(10, 6))
plt.plot(t_plot.numpy(), sir_pred[:, 0], label="S(t) Susceptible")
plt.plot(t_plot.numpy(), sir_pred[:, 1], label="I(t) Infected")
plt.plot(t_plot.numpy(), sir_pred[:, 2], label="R(t) Recovered")
plt.plot(t_plot.numpy(), sir_pred[:, 1] + sir_pred[:, 2], label="I+R (Cumulative)", linestyle="--")
plt.title("PINN SIR Prediction")
plt.xlabel("Normalized Time")
plt.ylabel("Proportion")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()


plt.figure(figsize=(10, 4))
plt.plot(loss_history)
plt.title("Training Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.grid(True)
plt.tight_layout()
plt.show()


i_r_plot_pred = sir_pred[:, 1] + sir_pred[:, 2]
i_r_plot_pred_denorm = i_r_plot_pred * confirmed.max()
plt.figure(figsize=(10, 6))
plt.plot(global_data["Date"], confirmed, label="Actual Confirmed", color="red")
plt.plot(global_data["Date"], i_r_plot_pred_denorm[:len(global_data["Date"])],
         label="Predicted I+R", linestyle="--", color="blue")
plt.title("Predicted vs Actual Confirmed Cases")
plt.xlabel("Date")
plt.ylabel("Confirmed Cases")
plt.xticks(rotation=45)
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
