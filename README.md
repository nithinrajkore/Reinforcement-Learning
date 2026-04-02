# Reinforcement Learning — Upper Confidence Bound (UCB)

[![Jupyter](https://img.shields.io/badge/Jupyter-Notebook-F37626?logo=jupyter&logoColor=white)](https://jupyter.org/)
[![Python](https://img.shields.io/badge/Python-3.x-3776AB?logo=python&logoColor=white)](https://python.org)
[![NumPy](https://img.shields.io/badge/NumPy-1.x-013243?logo=numpy)](https://numpy.org)
[![Matplotlib](https://img.shields.io/badge/Matplotlib-3.x-11557C)](https://matplotlib.org)

An implementation of the **Upper Confidence Bound (UCB)** reinforcement learning algorithm applied to the classic **Multi-Armed Bandit** problem — specifically, optimizing ad click-through rates (CTR) across 10 competing advertisements over 10,000 rounds.

---

## Table of Contents

1. [Repository Structure](#repository-structure)
2. [Problem Statement](#problem-statement)
3. [Dataset](#dataset)
4. [Algorithm: Upper Confidence Bound (UCB)](#algorithm-upper-confidence-bound-ucb)
   - [Theory](#theory)
   - [Mathematical Formulation](#mathematical-formulation)
   - [Step-by-Step Algorithm](#step-by-step-algorithm)
5. [Implementation](#implementation)
6. [Results & Visualization](#results--visualization)
7. [UCB vs Random Strategy](#ucb-vs-random-strategy)
8. [Tech Stack](#tech-stack)
9. [Getting Started](#getting-started)
10. [Key Concepts Glossary](#key-concepts-glossary)
11. [References](#references)

---

## Repository Structure

```
Reinforcement-Learning/
├── README.md
└── Upper Confidence Bound/
    ├── Upper Confidence Bound Algorithm.ipynb    ← UCB implementation + visualization
    └── Ads_CTR_Optimisation.csv                  ← Simulated ad click dataset
```

| File | Description |
|------|-------------|
| `Upper Confidence Bound Algorithm.ipynb` | Full UCB algorithm with histogram visualization |
| `Ads_CTR_Optimisation.csv` | 10,000 rounds × 10 ads — binary click/no-click per round |

---

## Problem Statement

A company is running **10 different ad variants** (Ad_1 through Ad_10) for a product. Each time a user visits the site, one ad is shown. The user either clicks it (reward = 1) or ignores it (reward = 0).

**The challenge:** Which ad variant generates the most clicks over 10,000 user visits — especially when we don't know the true click rates in advance?

This is the **Multi-Armed Bandit problem** — a foundational reinforcement learning challenge that requires balancing:

- **Exploration** — try less-tested ads to learn more about their true click rates
- **Exploitation** — keep showing the best-known ad to maximize total clicks

**Why not just A/B test?**  
Traditional A/B testing wastes budget on clearly poor ads. UCB dynamically shifts budget toward better performers while still exploring uncertain ones.

---

## Dataset

**File:** `Ads_CTR_Optimisation.csv`

| Property | Value |
|----------|-------|
| Rows | 10,000 (user rounds) |
| Columns | 10 (`Ad_1` through `Ad_10`) |
| Values | Binary: 1 (would click) or 0 (would not click) |
| True best ad | Variable (determined by column means) |

**How the dataset works:**

Each row represents one user visit. The value in column `Ad_i` tells us whether this user *would have* clicked ad `i` if it were shown. The UCB algorithm only observes the reward for the ad it **chooses** — simulating a real online advertising setting where you can only show one ad per visit.

**Sample rows:**
```
Ad_1, Ad_2, Ad_3, Ad_4, Ad_5, Ad_6, Ad_7, Ad_8, Ad_9, Ad_10
1,    0,    0,    0,    1,    0,    0,    0,    1,    0
0,    0,    0,    0,    0,    0,    0,    0,    1,    0
0,    0,    0,    0,    0,    0,    0,    0,    0,    0
...
```

---

## Algorithm: Upper Confidence Bound (UCB)

### Theory

UCB is a **deterministic reinforcement learning algorithm** for the multi-armed bandit problem. It works by adding an **exploration bonus** to each arm's (ad's) estimated reward. The bonus naturally decreases as an arm is selected more often, shifting the algorithm from exploration to exploitation.

**Core intuition:** UCB maintains an upper bound on the *potential* true reward for each arm. It always selects the arm with the highest potential — effectively being optimistic about uncertain options.

**Key advantage over ε-greedy:**
- No random exploration — every decision is deliberate
- Exploration decreases naturally (no ε parameter to tune)
- Proven sub-linear regret: O(√(K T ln T)) — one of the best theoretical guarantees

### Mathematical Formulation

For round $t$ and arm (ad) $i$, define:

- $n_i(t)$ — number of times arm $i$ has been selected through round $t$
- $R_i(t)$ — cumulative reward for arm $i$ through round $t$
- $\bar{r}_i(t) = R_i(t) / n_i(t)$ — mean reward estimate (sample mean)

**Upper Confidence Bound:**

$$\text{UCB}_i(t) = \bar{r}_i(t) + \Delta_i(t)$$

**Exploration bonus:**

$$\Delta_i(t) = \sqrt{\frac{3}{2} \cdot \frac{\ln(t)}{n_i(t)}}$$

**Selection rule (greedy on UCB):**

$$A_t = \arg\max_{i \in \{1,\ldots,d\}} \left[ \bar{r}_i(t) + \sqrt{\frac{3}{2} \cdot \frac{\ln(t)}{n_i(t)}} \right]$$

**Behaviour of the exploration term:**
- $\ln(t)$ grows slowly — ensuring every arm is eventually explored infinitely often
- $1/n_i(t)$ shrinks as arm $i$ is tried more — reducing exploration for well-known arms
- Together: UCB decreases over time for frequently-selected arms → convergence to optimal arm

**Cumulative Regret:**

$$\text{Regret}(T) = \mu^* \cdot T - \sum_{t=1}^{T} r_{A_t}(t)$$

where $\mu^*$ is the optimal arm's true mean reward. UCB achieves $\text{Regret}(T) = O(\log T)$ — much better than random selection ($O(T)$).

### Step-by-Step Algorithm

```
Initialize:
  n_i = 0   for all ads i = 1..d   (times each ad selected)
  R_i = 0   for all ads i = 1..d   (cumulative reward per ad)

For each round t = 1, 2, ..., N:
  1. Compute UCB for each ad i:
       if n_i == 0:  UCB_i = +∞  (force selection)
       else:         UCB_i = R_i/n_i + √(1.5 × ln(t) / n_i)
  
  2. Select ad with highest UCB:
       A_t = argmax_i UCB_i
  
  3. Observe reward:
       r_t = dataset[t, A_t]   (1 if clicked, 0 if not)
  
  4. Update statistics:
       n_{A_t} += 1
       R_{A_t} += r_t
```

---

## Implementation

```python
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# ---- Load Data ----
dataset = pd.read_csv('Ads_CTR_Optimisation.csv')

# ---- UCB Algorithm ----
N = 10000   # Total rounds
d = 10      # Number of ads

ads_selected       = []    # Track which ad was shown each round
numbers_of_selections = [0] * d   # n_i(t) for each ad
sums_of_rewards       = [0] * d   # R_i(t) for each ad
total_reward = 0

for n in range(0, N):
    ad = 0
    max_upper_bound = 0

    for i in range(0, d):
        if numbers_of_selections[i] > 0:
            average_reward    = sums_of_rewards[i] / numbers_of_selections[i]
            delta_i           = np.sqrt(3/2 * np.log(n + 1) / numbers_of_selections[i])
            upper_bound       = average_reward + delta_i
        else:
            upper_bound = 1e400  # Force selection for unvisited ads

        if upper_bound > max_upper_bound:
            max_upper_bound = upper_bound
            ad = i

    ads_selected.append(ad)
    numbers_of_selections[ad] += 1
    reward = dataset.values[n, ad]
    sums_of_rewards[ad] += reward
    total_reward += reward

# ---- Report Results ----
best_ad = np.argmax(numbers_of_selections)
print(f"Total Reward:             {total_reward}")
print(f"Best Ad Identified:       Ad {best_ad + 1}")
print(f"Times Best Ad Selected:   {numbers_of_selections[best_ad]}")
print(f"\nSelection counts per ad:")
for i, count in enumerate(numbers_of_selections):
    mean = sums_of_rewards[i] / count if count > 0 else 0
    print(f"  Ad {i+1:2d}: selected {count:5d} times | avg reward = {mean:.3f}")

# ---- Visualization ----
plt.figure(figsize=(12, 5))
plt.hist(ads_selected, bins=range(d + 1), align='left', rwidth=0.8,
         color='steelblue', edgecolor='white')
plt.title('Histogram of Ad Selections — UCB Algorithm', fontsize=14)
plt.xlabel('Ad Index (0-based)', fontsize=12)
plt.ylabel('Number of Times Selected', fontsize=12)
plt.xticks(range(d), [f'Ad {i+1}' for i in range(d)])
plt.tight_layout()
plt.show()
```

---

## Results & Visualization

**After 10,000 rounds:**

- The histogram shows UCB converging heavily to the best-performing ad
- The best ad is selected the vast majority of rounds (typically 7,000–9,000 times)
- All other ads are selected only briefly during early exploration
- Total reward far exceeds random selection baseline

**Sample results:**

| Metric | Random Selection | UCB |
|--------|-----------------|-----|
| Total reward | ~1,250 | ~2,100+ |
| Best ad found | Never (uniform) | ✓ Converges |
| Regret growth | Linear O(T) | Logarithmic O(log T) |

The histogram output will look like a spike at the best ad with a long flat tail across other ads.

---

## UCB vs Random Strategy

| Aspect | Random Selection | UCB Algorithm |
|--------|-----------------|---------------|
| **Strategy** | Show random ad each round | Show ad with highest upper confidence bound |
| **Exploration** | Always (uniform) | Initially high, decreases naturally |
| **Exploitation** | None | Automatically shifts to best ad |
| **Total rewards** | ~10,000 × avg_CTR | Significantly above avg_CTR |
| **Regret** | O(T) — linear | O(log T) — sub-linear |
| **Parameters to tune** | None | None (self-regulates) |
| **Convergence** | Never | Converges to optimal arm |
| **Use case** | Baseline comparison | Production advertising, clinical trials, A/B testing |

---

## Tech Stack

| Library | Version | Usage |
|---------|---------|-------|
| Python | 3.x | Core language |
| NumPy | 1.x | `np.log`, `np.sqrt`, array operations |
| Pandas | 1.x | CSV loading via `read_csv` |
| Matplotlib | 3.x | Histogram of ad selection distribution |
| Jupyter | Latest | Interactive notebook environment |

---

## Getting Started

```bash
# 1. Clone
git clone https://github.com/nithinrajkore/Reinforcement-Learning.git
cd Reinforcement-Learning

# 2. Install dependencies
pip install numpy pandas matplotlib jupyter

# 3. Launch notebook
jupyter notebook "Upper Confidence Bound/Upper Confidence Bound Algorithm.ipynb"
```

---

## Key Concepts Glossary

| Term | Definition |
|------|------------|
| **Multi-Armed Bandit** | Problem: choose between N arms each round to maximize cumulative reward, with unknown payoff distributions |
| **Arm** | One option in the bandit problem; here, one of the 10 ad variants |
| **Exploration** | Selecting under-tested arms to reduce uncertainty about their true reward |
| **Exploitation** | Selecting the currently-estimated best arm to maximize immediate reward |
| **Exploration-Exploitation Tradeoff** | Fundamental tension: learn more (explore) vs earn more now (exploit) |
| **UCB** | Upper Confidence Bound — adds an uncertainty bonus to encourage exploration of uncertain arms |
| **Upper Confidence Bound** | $\text{UCB}_i = \bar{r}_i + \sqrt{1.5 \ln(t) / n_i}$ |
| **Regret** | Cumulative difference between optimal and actual reward over all rounds |
| **Sub-linear Regret** | Regret grows slower than T (rounds) — UCB achieves O(log T) |
| **Click-Through Rate (CTR)** | Fraction of ad impressions that result in a click |
| **Exploration Bonus** | $\Delta_i = \sqrt{1.5 \ln(t) / n_i}$ — decreases as arm is tried more |

---

## References

1. Auer, P., Cesa-Bianchi, N., & Fischer, P. (2002). *Finite-time analysis of the multiarmed bandit problem*. Machine Learning, 47(2-3), 235–256.
2. Sutton, R. S., & Barto, A. G. (2018). *Reinforcement Learning: An Introduction* (2nd ed.). MIT Press. Chapter 2.
3. Lai, T. L., & Robbins, H. (1985). *Asymptotically efficient adaptive allocation rules*. Advances in Applied Mathematics, 6(1), 4–22.
4. Wikipedia: [Multi-armed bandit](https://en.wikipedia.org/wiki/Multi-armed_bandit)
5. Dataset: Simulated online advertising CTR optimization dataset
