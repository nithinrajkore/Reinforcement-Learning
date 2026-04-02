# 🎰 Reinforcement Learning — Upper Confidence Bound (UCB)

A Reinforcement Learning project implementing the **Upper Confidence Bound (UCB)** algorithm to solve the **Multi-Armed Bandit problem**, applied to ad click-through rate (CTR) optimization.

---

## 📌 Overview

This project demonstrates online reinforcement learning — specifically how an agent can intelligently explore and exploit options to maximize cumulative reward over time. The UCB algorithm is applied to a simulated advertising scenario where the goal is to identify the best-performing ad out of 10 candidates across 10,000 rounds, **without prior knowledge** of which ad performs best.


---

## 🧠 Problem Statement

A company wants to show users one of **10 different ads**. Each ad has an unknown probability of being clicked. The challenge is to figure out which ad performs best while **minimizing regret** — i.e., minimizing the number of times a suboptimal ad is shown.

This is the classic **Multi-Armed Bandit** problem.

---

## 📐 Algorithm — Upper Confidence Bound (UCB)

UCB balances **exploration** (trying less-tested ads) and **exploitation** (showing ads that have performed well so far) using the following formula:

For each ad $i$ at round $n$:

$$
\text{UCB}_i = \bar{x}_i + \sqrt{\frac{3}{2} \cdot \frac{\ln(n+1)}{N_i}}
$$

Where:
- $\bar{x}_i$ = average reward of ad $i$ so far
- $N_i$ = number of times ad $i$ has been selected
- $n$ = current round number

- If an ad has **never been selected**, it gets an upper bound of $\infty$ (ensuring it gets tried at least once)
- At each round, the ad with the **highest upper bound** is selected

---

## 📊 Dataset

`Ads_CTR_Optimisation.csv` contains a simulated dataset of **10,000 rounds × 10 ads**, where each cell is:
- `1` — the user **would have clicked** the ad
- `0` — the user **would not have clicked** the ad

> Note: In real-world UCB, only the reward of the *selected* ad is observed per round. The full dataset simulates the ground truth.

---

## 🔬 Implementation Steps

1. **Load Dataset** — Read `Ads_CTR_Optimisation.csv` using `pandas`
2. **Initialize** — Set `N = 10000` rounds, `d = 10` ads; track `number_of_selections`, `sums_of_rewards`, `total_rewards`
3. **Run UCB Loop** — For each of 10,000 rounds:
   - Compute the UCB score for each ad
   - Select the ad with the maximum upper bound
   - Observe reward and update counters
4. **Visualize** — Plot a histogram of ad selection frequency to identify the best ad

---

## 📈 Results

The histogram of ad selections clearly shows that the UCB algorithm converges on the **best-performing ad** (Ad 5), selecting it far more frequently than the others after an initial exploration phase.

---

## 🛠️ Tech Stack

- **Language:** Python 3.10
- **Libraries:** `numpy`, `pandas`, `matplotlib`, `math`
- **Algorithm:** Upper Confidence Bound (UCB1 variant)

---

## 🚀 Getting Started

### Prerequisites
```bash
pip install numpy pandas matplotlib
```

## Run the Notebook

1. Clone the repository:
```bash
git clone https://github.com/nithinrajkore/Reinforcement-Learning.git
cd "Reinforcement-Learning/Upper Confidence Bound"
```

2. Launch Jupyter Notebook:
```bash
jupyter notebook "Upper Confidence Bound Algorithm.ipynb"
```

3. Run all cells top to bottom.

👤 Author

Nithin Raj Kore

GitHub
