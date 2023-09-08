# Introduction

Status: Done

- Brownian motion is the building block of stochastic calculus and therefore, the key to simulating stochastic processes.
- **Wiener processes**, the other name given to Brownian motions, can be used to build processes with different properties and behaviors.
- In simple terms, **Brownian motion is a continuous process such that its increments for any time scale are drawn from a normal distribution.**

## ****Definition****

**Def.** A standard (one-dimensional) Wiener process (also called Brownian motion) is
a stochastic process $\{W_t\}_{t≥0+}$ indexed by nonnegative real numbers t with the following
properties:
(1) $W_0 = 0$.
(2) With probability 1, the function $t → W_t$ is continuous in $t$.
(3) The process $\{W_t\}_{t≥0}$ has stationary, independent increments.
(4) The increment $W_{t+s} - W_s$ has the $N(0, t)$ distribution.

for all $0 = t_0 < t_1 < … < t_m$ **the increments

$*W(t_1) — W(t_0), W(t_2) — W(t_1), …, W(t_m) — W(t_{m-1})*$

are independent and normally distributed. The mean of the distribution (normal) is zero and its variance is the time difference $*t_{i+1} — t_i*$.

### Generating a Brownian motion in Python

The following code generates the increments of a Wiener process ($*dW*$) discretely sampled in unit time as well as the process path ($*W*$):

```python
from typing import Optional

import numpy as np

def get_dW(T: int, random_state: Optional[int] = None) -> np.ndarray:
    """
    Sample T times from a normal distribution,
    to simulate discrete increments (dW) of a Brownian Motion.
    Optional random_state to reproduce results.
    """
    np.random.seed(random_state)
    return np.random.normal(0.0, 1.0, T)

def get_W(T: int, random_state: Optional[int] = None) -> np.ndarray:
    """
    Simulate a Brownian motion discretely samplet at unit time increments.
    Returns the cumulative sum
    """
    dW = get_dW(T, random_state)
    # cumulative sum and then make the first index 0.
    dW_cs = dW.cumsum()
    return np.insert(dW_cs, 0, 0)[:-1]
```

An example, for $*T*$ (sample size) of 1,000:

```python
dW = get_dW(T=1_000)
W = get_W(T=1_000)

#----------------------------------------------------------------
# plot

import matplotlib.pyplot as plt 
import seaborn as sns

fig = plt.figure(figsize=(15, 5))

title = "Brownian motion increments"
plt.subplot(1, 2, 1)
plt.plot(dW)
plt.gca().set_title(title, fontsize=15)
plt.xticks(fontsize=15)
plt.yticks(fontsize=15)

title = "Brownian motion path"
plt.subplot(1, 2, 2)
plt.plot(W)
plt.gca().set_title(title, fontsize=15)
plt.xticks(fontsize=15)
plt.yticks(fontsize=15)
```

![Figure_1.png](Figure_1.png)

## ****Correlated processes****

Brownian motion can be correlated to another Brownian motion.

Let $*W_1*$ be a Brownian motion and $*W_3*$ another Brownian motion correlated to $*W_1*$, then:

$$
dW_{3,t} = \rho dW_{1,t} + \sqrt{1-\rho^2} dW_{2,t}
$$

where $*W_2*$ is another independent Brownian motion. The correlation of $*W_3*$ and $*W_1*$ is $ρ$.

**Note:**  even though there is correlation between the two processes $*W_3*$ and $*W_1*$, there are still two sources of randomness, $*W_1*$ and $*W_2*$., i.e. , **correlation does not decrease the sources of randomness.**

The following function generates a correlated Brownian motion. Returns the increments of such correlated process.

```python
from typing import Optional

import numpy as np

def _get_correlated_dW(
    dW: np.ndarray, rho: float, random_state: Optional[int] = None
) -> np.ndarray:
    """
    Sample correlated discrete Brownian increments to given increments dW.
    """
    dW2 = get_dW(
        len(dW), random_state=random_state
    )  # generate Brownian icrements.
    if np.array_equal(dW2, dW):
        # dW cannot be equal to dW2.
        raise ValueError(
            "Brownian Increment error, try choosing different random state."
        )
    return rho * dW + np.sqrt(1 - rho ** 2) * dW2
```

However, we seldom want just a pair of correlated processes. Rather, we often require many process somehow correlated, an N-dimensional Wiener process.

The following algorithm’s idea is to first generate one Brownian motion, then, another correlated to the first one by $ρ$, the subsequent processes should be correlated by $ρ$  to a random choice of the processes already generated.

```python
from typing import Optional

import numpy as np

def get_corr_dW_matrix(
    T: int,
    n_procs: int,
    rho: Optional[float] = None,
    random_state: Optional[int] = None,
) -> np.ndarray:
    """
    2D array of n_procs discrete Brownian Motion increments dW.
    Each column of the array is one process.
    So that the resulting shape of the array is (T, n_procs).
        - T is the number of samples of each process.
        - The correlation constant rho is used to generate a new process,
            which has rho correlation to a random process already generated,
            hence rho is only an approximation to the pairwise correlation.
        - Optional random_state to reproduce results.
    """
    rng = np.random.default_rng(random_state)
    dWs: list[np.ndarray] = []
    for i in range(n_procs):
        random_state_i = _get_random_state_i(random_state, i)
        if i == 0 or rho is None:
            dW_i = get_dW(T, random_state=random_state_i)
        else:
            dW_corr_ref = _get_corr_ref_dW(dWs, i, rng)
            dW_i = _get_correlated_dW(dW_corr_ref, rho, random_state_i)
        dWs.append(dW_i)
    return np.asarray(dWs).T
    

def _get_random_state_i(random_state: Optional[int], i: int) -> Optional[int]:
    """Add i to random_state is is int, else return None."""
    return random_state if random_state is None else random_state + i

def _get_corr_ref_dW(
    dWs: list[np.ndarray], i: int, rng: np.random.Generator
) -> np.ndarray:
    """
    Choose randomly a process (dW) the from the
    already generated processes (dWs).
    """
    random_proc_idx = rng.choice(i)
    return dWs[random_proc_idx]
```

Using this code we can generate as many processes as we wish:

```python
T = 1_000
n_procs = 53
rho = 0.99

corr_dWs = get_corr_dW_matrix(T, n_procs, rho)

#----------------------------------------------------------------
# plot

import matplotlib.pyplot as plt 
import seaborn as sns

fig = plt.figure(figsize=(15, 5))

# paths
title = "Correlated Brownian motion paths"
plt.subplot(1, 2, 1)
plt.plot(np.cumsum(corr_dWs, axis=0))
plt.gca().set_title(title, fontsize=15)
plt.xticks(fontsize=15)
plt.yticks(fontsize=15)

# correlation
title = "Correlation matrix heatmap"
plt.subplot(1, 2, 2)
sns.heatmap(np.corrcoef(corr_dWs, rowvar=False), cmap="viridis")
plt.gca().set_title(title, fontsize=15)
plt.xticks(fontsize=15)
plt.yticks(fontsize=15)
```

![Figure_2.png](Figure_2.png)

## ****Diffusive processes****

The most general form in which we can use a Brownian motion to build more complex processes is the Itô diffusive process

$$
dX_t = a_1(X_t,t)dt + b_1(X_t,t)dW_t
$$

where $*a_1*$ and $*b_1*$ are functions of $*t$* (time) and the process itself. The first term corresponds to the deterministic part and the second term to the random part.

**Note**: the Brownian motions in such diffusive processes can be correlated, the same way as in the previous section. If the Brownian Motions ($*W_t*$) are correlated, then the Itô processes ($*X_t*$) are correlated.
