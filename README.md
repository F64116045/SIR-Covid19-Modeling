#  SIR-Covid19 Model Simulation and Comparison

This project simulates and compares the early global spread of COVID-19 using the **SIR (Susceptible-Infectious-Recovered) epidemiological model**, with real-world data to verify model performance.

---

##  Dataset Description

The dataset used comes from a public COVID-19 dataset on Kaggle, covering **187 countries and regions** from **January 22, 2020 to July 22, 2020**.

| Column Name      | Description                         |
|------------------|-------------------------------------|
| `Country/Region` | Name of the country or region       |
| `New cases`      | Daily confirmed new cases           |
| `New deaths`     | Daily new deaths                    |
| `New recovered`  | Daily new recoveries                |



---

##   Goals

The main goals of this project are:

- Apply the **SIR model** to simulate the spread of COVID-19  
- Explore and compare two different methods for setting initial conditions  
- Evaluate the model’s performance in approximating the early-stage pandemic spread

---

##  Model Methods and Setup

This project explores two approaches for initializing the model:

### Method 1: Estimate initial values from data

Using the earliest available global data (2020-01-24):

- Confirmed global cases on that day: 555  
- Assume the global population is approximately 7.8 billion  
- Calculated as:
  - `I(0) = 555 / 7.8e9 ≈ 7.12e-8`
  - `S(0) = 1 - I(0) ≈ 0.9999999288`
  - `R(0) = 0`

### Method 2: Use custom initial assumptions

Alternatively, define a set of initial conditions (e.g., assumed infection ratio, recovery rate, basic reproduction number \( R_0 \)) for comparison and sensitivity analysis.

---

##  Tools & Libraries Used

- **Python** (NumPy, Matplotlib, SciPy)
- **Jupyter Notebook**
- **Pandas** for data cleaning and processing
- **ydata-profiling** (formerly pandas-profiling) for automated data reports

---


