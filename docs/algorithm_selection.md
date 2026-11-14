# Algorithm selection (plan P6.1-P6.5)

This document fixes the algorithm comparison *before* any learning curve is
produced.  It covers the two tasks in the project:

* **FC** - the flight-controller rate-tracking task (`interceptor/fc/env.py`),
  action dimension 4, dense quadratic reward, episode 2 s.
* **G3** - the quadcopter guidance task in arm A3 (`interceptor/guidance/env.py`,
  `mode="apn_rl"`), action dimension 3, potential-based reward plus a sparse
  terminal kill bonus, episode up to 18 s (900 decisions).

The candidates are PPO, SAC and TD3, all from Stable-Baselines3 2.9.  The
comparison uses the same MLP widths, the same environment-step budget and
symmetric (non-privileged) policies.  The privileged critic used by the final
guidance models is a PPO-specific extension and is *not* part of the fairness
comparison; it is evaluated separately in P7.4.

## 1. Common setting

| Item | Value |
|---|---|
| Observation | task-specific; FC actor 33, guidance actor 85 |
| Action | FC 4 motors in [-1, 1]; guidance 3 residuals in [-1, 1] |
| Network | `pi` and `vf`/`qf` MLPs `[256, 256, 128]`, ReLU. Project choice carried over from the v1 notebooks; closest published networks are 2 x 256 ReLU (SAC, TD3 reference code) and 2 x 64 tanh (PPO). Defended by the `network` ablation in `configs/ablations.yaml` (Andrychowicz et al. 2021, arXiv:2006.05990, Sec. 3.2, pp. 5-6: the optimal policy width depends on the environment; tanh beat ReLU) |
| Discount | 0.99 for flight-controller runs; every guidance run, including the P6.4 comparison, uses the shaping discount 0.99791 (`training/runner.py`, BUGS #26) |
| Seeds | 0, 1, 2 |
| Budget | equal environment steps, `budgets.algorithm_selection_steps` in `configs/training.yaml` |
| Evaluation | fixed, disjoint seeds; 100 episodes per cell (§6 of the implementation plan) |
| Hyperparameters | Stable-Baselines3 2.9.0 library defaults (file and line recorded per value), which equal PPO App. A Table 3 / Table 1 (Schulman et al. 2017), SAC App. D Table 1 (Haarnoja et al. 2018) and TD3 Sec. 6.1 / App. C Table 3 (Fujimoto et al. 2018). TD3 exploration noise N(0, 0.1) from Fujimoto et al. Sec. 6.1 (SB3 adds none by default; BUGS #31). Project choice: TD3 learning rate 3e-4 (paper and SB3 1e-3; the authors' code github.com/sfujim/TD3 uses 3e-4, and it equals the PPO/SAC step size). Every line of `configs/training.yaml` names its source; `tests/unit/test_algos.py` checks every value against the installed SB3 |

The constructor is `training/algos.py::build_model`, so every algorithm gets
the identical network and environment wrappers.  Any change to the comparison
must be made there and in `configs/training.yaml`.

## 2. PPO

### 2.1 Policy gradient and advantage

For a stochastic policy $\pi_\theta(a\mid s)$ the policy-gradient estimator is

$$
\nabla_\theta J(\theta)
= \mathbb{E}\left[\nabla_\theta \log \pi_\theta(a_t\mid s_t)\, \hat A_t\right],
$$

with $\hat A_t$ an advantage estimate.  Subtracting a state-dependent baseline
$V_\phi(s_t)$ does not bias the gradient and reduces variance.  Generalized
advantage estimation (GAE; Schulman et al., 2016) forms

$$
\hat A_t = \sum_{l=0}^{T-t-1} (\gamma\lambda)^l \delta_{t+l},
\qquad
\delta_t = r_t + \gamma V_\phi(s_{t+1}) - V_\phi(s_t),
$$

which trades bias for variance through $\lambda\in[0,1]$; the implementation
uses $\lambda=0.95$.

### 2.2 Trust region and the clipped objective

TRPO (Schulman et al., 2015) maximizes the surrogate

$$
L^{\text{TRPO}}(\theta)
= \mathbb{E}\left[\frac{\pi_\theta(a\mid s)}{\pi_{\theta_{\text{old}}}(a\mid s)}
  \hat A\right]
$$

subject to a KL constraint $\mathbb{E}[\mathrm{KL}(\pi_{\theta_{\text{old}}}\|\pi_\theta)]\le\delta$.
PPO replaces the constrained problem with the clipped surrogate

$$
L^{\text{CLIP}}(\theta)
= \mathbb{E}\left[\min\left(
  \rho_t(\theta)\hat A_t,\;
  \operatorname{clip}(\rho_t(\theta), 1-\epsilon, 1+\epsilon)\hat A_t
\right)\right],
\qquad
\rho_t(\theta)=\frac{\pi_\theta(a_t\mid s_t)}{\pi_{\theta_{\text{old}}}(a_t\mid s_t)},
$$

with $\epsilon=0.2$.  The minimum removes the incentive to move the ratio
outside the trust region in the improving direction, giving a first-order
approximation to a KL trust region at the cost of one extra clipping operation.
The full loss adds a value regression and an entropy term,

$$
L = -L^{\text{CLIP}} + c_v \mathbb{E}\left[(V_\phi(s_t)-\hat R_t)^2\right]
    - c_e \mathbb{E}\left[\mathcal H(\pi_\theta(\cdot\mid s_t))\right],
$$

with $c_v=0.5$ (SB3 default) and $c_e=0$ (Schulman et al. 2017, Sec. 6.1: no entropy bonus).

PPO is on-policy: each batch of `n_steps` is used for `n_epochs` passes and then
discarded.  It therefore needs no replay buffer and handles a non-stationary
reward (the moving target) without stale off-policy data.  Its weakness is
sample efficiency on long-horizon tasks.

## 3. TD3

TD3 (Fujimoto et al., 2018) learns a deterministic actor
$\mu_\theta(s)$ and two critics $Q_{\phi_1}, Q_{\phi_2}$.

### 3.1 Deterministic policy gradient

The deterministic policy gradient (Silver et al., 2014) is

$$
\nabla_\theta J(\theta)
= \mathbb{E}_{s\sim\mathcal D}\left[
  \nabla_\theta \mu_\theta(s)\,
  \nabla_a Q_{\phi_1}(s,a)\big|_{a=\mu_\theta(s)}
\right].
$$

### 3.2 Clipped double-Q target

The critic target uses the *minimum* of the two target critics,

$$
y = r + \gamma(1-d)\min_{i=1,2} Q_{\phi_i'}\left(s',\,
  \operatorname{clip}\left(\mu_{\theta'}(s')+\operatorname{clip}(\epsilon,-c,c),
  a_{\min},a_{\max}\right)\right),
\qquad \epsilon\sim\mathcal N(0,\sigma^2),
$$

with target policy smoothing $\sigma=0.2$, $c=0.5$.  Taking the minimum
counteracts the positive bias of the max in Q-learning; smoothing prevents the
critic from exploiting narrow peaks in the target policy.

### 3.3 Delayed updates

The actor and target networks are updated every `policy_delay=2` critic steps,
and the targets move polyak-averaged with $\tau=0.005$:

$$
\phi_i' \leftarrow \tau\phi_i + (1-\tau)\phi_i',
\qquad
\theta' \leftarrow \tau\theta + (1-\tau)\theta'.
$$

TD3 is off-policy and replay-efficient, but the deterministic actor explores
through added Gaussian noise and can be brittle with a sparse terminal reward.
It also gives no natural entropy bonus.

## 4. SAC

SAC (Haarnoja et al., 2018) maximizes the maximum-entropy objective

$$
J(\theta)=\mathbb{E}_{\pi_\theta}\left[
  \sum_t \gamma^t\left(r_t + \alpha\,\mathcal H(\pi_\theta(\cdot\mid s_t))\right)
\right],
$$

which yields the soft Bellman equation

$$
Q^\pi(s,a) = r(s,a) + \gamma\,\mathbb{E}_{s'\sim p}\left[
  V^\pi(s')\right],
\qquad
V^\pi(s)=\mathbb{E}_{a\sim\pi}\left[Q^\pi(s,a) - \alpha\log\pi(a\mid s)\right].
$$

The critic is trained by soft Bellman residual regression,

$$
L_Q(\phi)=\mathbb{E}_{(s,a,r,s')\sim\mathcal D}\left[
  \left(Q_\phi(s,a) - y\right)^2\right],
\qquad
y = r + \gamma(1-d)\left(\min_{i=1,2}Q_{\phi_i'}(s',a') - \alpha\log\pi_\theta(a'\mid s')\right),
\quad a'\sim\pi_\theta(\cdot\mid s'),
$$

and the actor by

$$
L_\pi(\theta)=\mathbb{E}_{s\sim\mathcal D,\,a\sim\pi_\theta}
\left[\alpha\log\pi_\theta(a\mid s) - Q_\phi(s,a)\right].
$$

The temperature $\alpha$ is tuned automatically against a target entropy
(`ent_coef="auto"`).  SAC is off-policy and sample-efficient; the entropy term
encourages exploration and action smoothness.  It has two critics, a stochastic
actor and a replay buffer, so it is the most expensive per environment step.

## 5. Criteria table

Scores are 1 (poor) to 5 (excellent) for *this* project; the rationale is in
the notes.  The table is deliberately written before the runs so the decision
rule in §6 cannot be fitted to the results.

| Criterion | PPO | SAC | TD3 | Rationale |
|---|---:|---:|---:|---|
| Sample efficiency (FC, dense reward) | 3 | 5 | 4 | off-policy replay helps the 2 s dense task |
| Sample efficiency (G3, sparse terminal bonus) | 4 | 4 | 2 | TD3 exploration is weakest with sparse rewards |
| Training stability across seeds | 5 | 3 | 3 | PPO's clipped update is the most predictable |
| Long-horizon credit assignment (900 steps) | 4 | 4 | 3 | GAE and soft values both propagate; TD3 target smoothing slows it |
| Action smoothness | 3 | 4 | 4 | entropy / deterministic actor smooth the policy |
| Wall-clock per environment step | 5 | 2 | 3 | PPO has no replay and one critic |
| Hyperparameter sensitivity | 4 | 3 | 2 | TD3 has coupled delay/target-noise settings |
| Asymmetric critic support in SB3 | 5 | 2 | 2 | a custom PPO policy is the documented path; off-policy variants need more surgery |
| Implementation risk | 5 | 3 | 3 | one network vs actor + two critics + targets |
| **Total** | **38** | **30** | **27** | |

## 6. Decision rule (written before the runs)

**Rule.** The analytically preferred algorithm is adopted unless another
algorithm beats it on final performance with **non-overlapping 95% confidence
intervals**.  "Final performance" means the protocol metric for the task: mean
kill rate (Wilson interval) for G3 and mean episode reward (seed mean with
range) for FC.  Ties are broken by area under the learning curve, then by
wall-clock time.  A result is only admissible if all three seeds completed the
same environment-step budget.

If the non-overlapping condition is not met, the analytic preference stands.

**Analytic preference: PPO.**  Reasons, in order:

1. The final guidance model needs an asymmetric actor-critic: the actor sees 85
   observations, the critic additionally sees 16 privileged values.  This is a
   first-class extension in SB3's PPO (`AsymmetricActorCriticPolicy`), whereas
   SAC/TD3 would require custom actor/critic classes and target networks.
2. A 900-step episode with a terminal kill bonus rewards stable on-policy
   updates and GAE more than replay efficiency; PPO's variance is lower across
   seeds in the criteria table.
3. The flight-controller task is short and dense; if SAC wins there, the FC
   model can be trained with SAC independently, because the guidance decision
   only constrains the guidance stack.

## 7. Outcome (P6.5)

The 18-run comparison (3 algorithms x 2 tasks x 3 seeds) is notebook
`notebooks/02_algorithm_selection.ipynb`.  It records learning curves, final
performance, area under the curve, wall-clock, seed variance and action
smoothness.  The executed results are attached to the release; this section is
updated with the table and the final decision after that run.

**Provisional decision:** PPO, by the analytic preference in §6.  The
provisional status is removed only when the notebook's confidence intervals
either confirm PPO or satisfy the non-overlap condition for a challenger.

## References

* Schulman, J., Levine, S., Abbeel, P., Jordan, M., Moritz, P. (2015).
  Trust Region Policy Optimization. ICML.
* Schulman, J., Moritz, P., Levine, S., Jordan, M., Abbeel, P. (2016). High-
  Dimensional Continuous Control Using Generalized Advantage Estimation. ICLR.
* Schulman, J., Wolski, F., Dhariwal, P., Radford, A., Klimov, O. (2017).
  Proximal Policy Optimization Algorithms. arXiv:1707.06347.
* Silver, D., Lever, G., Heess, N., Degris, T., Wierstra, D., Riedmiller, M.
  (2014). Deterministic Policy Gradient Algorithms. ICML.
* Fujimoto, S., van Hoof, H., Meger, D. (2018). Addressing Function
  Approximation Error in Actor-Critic Methods. ICML.
* Haarnoja, T., Zhou, A., Abbeel, P., Levine, S. (2018). Soft Actor-Critic:
  Off-Policy Maximum Entropy Deep Reinforcement Learning with a Stochastic
  Actor. ICML.
* Raffin, A., Hill, A., Gleave, A., Kanervisto, A., Ernestus, M., Dormann, N.
  (2021). Stable-Baselines3: Reliable Reinforcement Learning Implementations.
  JMLR 22(268).
