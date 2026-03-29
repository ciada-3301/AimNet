<div align="center">

```
    █████╗ ██╗███╗   ███╗███╗   ██╗███████╗████████╗
   ██╔══██╗██║████╗ ████║████╗  ██║██╔════╝╚══██╔══╝
███████║██║██╔████╔██║██╔██╗ ██║█████╗     ██║
██╔══██║██║██║╚██╔╝██║██║╚██╗██║██╔══╝     ██║
██║  ██║██║██║ ╚═╝ ██║██║ ╚████║███████╗   ██║
╚═╝  ╚═╝╚═╝╚═╝     ╚═╝╚═╝  ╚═══╝╚══════╝   ╚═╝
```

**A neural network that learns to aim a projectile — trained entirely from physics.**

[![Python](https://img.shields.io/badge/Python-3.8+-3776ab?style=flat-square&logo=python&logoColor=white)](https://python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c?style=flat-square&logo=pytorch&logoColor=white)](https://pytorch.org)
[![License: MIT](https://img.shields.io/badge/License-MIT-22c55e?style=flat-square)](LICENSE)
[![Tkinter](https://img.shields.io/badge/UI-Tkinter-fbbf24?style=flat-square)]()

</div>

---

## What is this?

AimNet is a small supervised learning project that teaches a neural network to solve a projectile motion problem — given a target position, predict the exact firing angle needed to hit it.

What makes it interesting is the training approach: instead of collecting data from trial and error (reinforcement learning), we use the **closed-form physics equation** to generate perfect labels. The network learns to approximate the inverse of a formula it never directly sees.

After training, a **ballistic targeting terminal** opens — a retro phosphor-green interface where you drag the cannon and target anywhere on the field and watch the AI fire its shot in real time.

```
                     ✦ TARGET
                    /
           . · ˜ ˜
         .
       .
    ◈ CANNON
```

---

## The journey to get here

This project went through three distinct phases — each one teaching a real lesson about machine learning:

### Phase 1 — Naive REINFORCE (didn't work)

The first attempt used a simple policy gradient loop: fire a shot, observe how close it got, use that as the reward signal. After 20,000 episodes the model was still stuck at `-100` reward every step.

**Why it failed:** With random initialisation, a random angle hits a random target roughly 2% of the time. That means 98% of gradient steps came from misses, and the rare hits were drowned out. The model never accumulated enough positive signal to bootstrap from.

### Phase 2 — Proper REINFORCE with baseline (still didn't work)

Added batched episodes, advantage normalisation, entropy bonuses, and a running baseline — all the standard tricks. Still flat. Mean reward hovering around `-35` for 800 updates.

**Why it failed:** REINFORCE is fundamentally high-variance. Without a way to generate *dense* signal (a correct answer for every sample), the noise overwhelmed the learning. The reward landscape had no gradient to follow.

### Phase 3 — Supervised pretraining (worked immediately)

The key insight: **we already have the answer**. Projectile motion has an exact closed-form solution. We can compute the optimal angle analytically and use it as a training label. This gives the network a perfect teacher for every single training example.

After 8,000 steps of supervised learning, MSE loss dropped from `0.1138` to `0.0000010` — a 99,990× reduction, corresponding to a mean angular error of under 0.05°.

---

## Architecture

```
Input (3)  →  Hidden (256)  →  Hidden (256)  →  Hidden (256)  →  Output head (1)  →  sigmoid × π/2  →  θ
```

| Layer | Size | Activation | Notes |
|---|---|---|---|
| Input | 3 | — | dx/100, dy/30, v/50 |
| Hidden × 3 | 256 | ReLU | `nn.Linear` + `F.relu` |
| Output head | 1 | Sigmoid × π/2 | Constrains output to [0, π/2] |

**Why 3 inputs?** Projectile physics is fully determined by three numbers: horizontal distance to target, vertical height of target, and muzzle velocity. Everything else cancels out. The divisions (`/100`, `/30`, `/50`) normalise each value to roughly [0, 1], which stabilises gradient magnitudes during training.

**Why 256 neurons?** The relationship between (dx, dy) and θ is a smooth nonlinear curve. 256 neurons gives the network plenty of capacity without any risk of underfitting, and at this scale training is still fast (< 60 seconds on CPU).

---

## Training

The training loop generates batches of random reachable targets, computes the analytical optimal angle for each, and minimises MSE between the network's prediction and that label.

```python
# Core training loop (simplified)
for step in range(8000):
    # Generate a batch of reachable targets
    while len(batch) < 256:
        xt = random() * 90 + 10   # x in [10, 100]
        yt = random() * 28        # y in [0,  28]
        theta_opt = analytical_theta(xt, yt)
        if theta_opt is not None:
            batch.append((make_state(xt, yt), theta_opt))

    # Supervised step
    pred  = model(states)
    loss  = F.mse_loss(pred, targets)
    loss.backward()
    optimizer.step()
```

The analytical solution used to generate labels:

```
y = x·tan(θ) - (g·x²) / (2v²·cos²θ)

Rearranges to a quadratic in tan(θ):
a·tan²(θ) + b·tan(θ) + c = 0

where a = gx²/2v²,  b = -x,  c = y + gx²/2v²
```

We take the smaller (flatter) positive root — the low-angle trajectory.

### Training curve

```
Step      0  |  MSE: 0.11380  |  err: ~18.5°
Step    500  |  MSE: 0.00009  |  err: ~1.7°
Step   1000  |  MSE: 0.00003  |  err: ~1.0°
Step   2000  |  MSE: 0.000002 |  err: ~0.25°
Step   4000  |  MSE: 0.000001 |  err: ~0.18°   ← converged
```

---

## The visualizer

Once training completes, a targeting terminal opens automatically.

```
┌─────────────────────────────────────────────────────────────────────┐
│  ▶  AIMNET BALLISTIC TARGETING SYSTEM  ◀         v=40m/s  g=9.8m/s² │
├──────────────────────────────────────────────────────────┬──────────┤
│                                                          │ TARGETING│
│   10  20  30  40  50  60  70  80  90  100  110  120     │ DATA     │
│                                                  ✦       │──────────│
│ 30                         . ˜ ˜ ˜               TARGET  │ CANNON   │
│ 20             . ˜ ˜ ˜ ˜                                 │ (10, 0)  │
│ 10    . ˜ ˜ ˜                                            │ TARGET   │
│  0  ◈──────────────────────────────────────────────────  │ (90, 25) │
│     CANNON                                               │──────────│
│                                                          │ θ PRED   │
│                                                          │  28.4°   │
│                                                          │──────────│
│                                              [ FIRE ]    │ HIT RATE │
└──────────────────────────────────────────────────────────┴──────────┘
```

**Controls:**
- Drag `◈` to reposition the cannon
- Drag `✦` to reposition the target
- The ghost trajectory (faint) shows the true optimal arc
- Click `[ FIRE ]` to launch — the AI predicts θ, the barrel snaps to that angle, and the projectile animates along the arc
- The sidebar shows predicted θ, optimal θ, the error between them, and miss distance

---

## Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/aimnet.git
cd aimnet

# Install dependencies (PyTorch + standard library only)
pip install torch

# Run — training window opens first, terminal opens automatically after
python aimnet_with_visualizer.py
```

**Requirements:**
- Python 3.8+
- PyTorch 2.0+ (CPU is fine — trains in under 60 seconds)
- `tkinter` — ships with Python on Windows and macOS. On Linux: `sudo apt install python3-tk`

No GPU needed. No other dependencies.

---

## Project structure

```
aimnet/
├── aimnet_with_visualizer.py   # Main file — training + UI
│
├── ai-games/                   # Companion web games (teach AI concepts)
│   ├── index.html              # Hub page
│   ├── game1-neural-network.html
│   ├── game2-train-test.html
│   ├── game3-overfitting.html
│   └── game4-image-recognition.html
│
└── loss_explainer.html         # Interactive loss curve explorer
```

---

## Companion web games

The `ai-games/` folder contains four standalone HTML games designed to teach the concepts behind this project to anyone new to machine learning. Open `ai-games/index.html` in any browser — no installation needed.

| Game | Concept |
|---|---|
| 🧠 Neural network decisions | Drag sliders to control a spam classifier in real time |
| 🔬 Training vs testing data | Build a training set for an animal classifier, then quiz it |
| 📈 Overfitting | Watch a model memorise vs generalise as complexity increases |
| 👁️ Image recognition | Draw on a pixel grid and see how the AI converts it to features |

---

## Key lessons from this project

**Reinforcement learning is hard to bootstrap.** When your environment gives a positive signal only 2% of the time, gradient descent has almost nothing to work with. RL shines when you genuinely can't generate labels any other way.

**Use your domain knowledge.** If you can write down the answer — even partially, even approximately — supervised learning will beat RL for that part of the problem every time. RL should be a last resort, not a first instinct.

**Normalise your inputs.** The difference between `dx = 70` and `dy = 0.8` feeding into the same layer caused unstable gradients in early experiments. Dividing by the expected maximum range resolved it immediately.

**The train/test split exists for a reason.** An early version evaluated the model on targets that included unreachable positions (discriminant < 0). Hit rate looked terrible even though the model was actually near-perfect — the denominator was wrong.

---

## Possible extensions

- **Variable muzzle velocity** — randomise `v` during training and add it as a meaningful third input
- **Wind resistance** — modify the physics simulator and retrain; the network architecture stays identical
- **Multiple targets** — extend to a sequence prediction problem
- **RL fine-tuning** — now that the model has a good initialisation from supervised learning, a small RL phase on the actual simulator could squeeze out the last errors from physics approximations

---

<div align="center">

Built iteratively, starting from a broken REINFORCE loop and ending at a near-perfect supervised learner.
The mistakes were the interesting part.

</div>
