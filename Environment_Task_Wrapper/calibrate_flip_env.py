"""
calibrate_flip_env.py — Section 1 sanity + normalisation calibration.

Run this ONCE on your machine to:
  * confirm the environment steps at all (checks the numpy<2 requirement),
  * verify the flip config (tilt termination disabled, altitude termination kept),
  * measure the raw per-field observation ranges under hover-centred exploration,
  * suggest / sanity-check the fixed normalisation scales used in flip_env.ObsConfig.

Usage (from inside the repo's Simulation/ folder, or with it on PYTHONPATH):
    python calibrate_flip_env.py
"""
import numpy as np
import warnings
warnings.filterwarnings("ignore")

from quad_velocity_env import QuadcopterVelocityEnv

RAW_NAMES = ["tvx", "tvy", "tvyaw", "x", "y", "z", "q0", "q1", "q2", "q3",
             "vx", "vy", "vz", "p", "q", "r", "w1", "w2", "w3", "w4"]


def main(n_episodes: int = 30, noise_std: float = 0.35, seed0: int = 0):
    if int(np.__version__.split(".")[0]) >= 2:
        print(f"[WARN] numpy {np.__version__} detected. The bobzwik dynamics break on "
              f"numpy 2.x — install numpy<2 (and scipy<1.13) before training.")

    env = QuadcopterVelocityEnv(obs_mode="full", terminate_on_unstable=True,
                                max_tilt_rad=np.inf, max_abs_z=5.0)
    hover_a = 2 * (env.hover_w - env.min_w) / (env.max_w - env.min_w) - 1.0
    print(f"motor speeds: min={env.min_w:.1f} max={env.max_w:.1f} hover={env.hover_w:.1f} "
          f"(hover action ~ {hover_a:+.4f})")

    # 1) upright reset check
    obs, _ = env.reset(seed=seed0)
    from flip_env import quat_to_projected_gravity
    g = quat_to_projected_gravity(env.quad.quat)
    print(f"reset: pos={np.round(env.quad.pos,3)} quat={np.round(env.quad.quat,3)} "
          f"projected_gravity={np.round(g,3)} (upright expects [0,0,1])")

    # 2) pure-hover stability
    obs, _ = env.reset(seed=seed0)
    for t in range(env.max_steps):
        obs, r, te, tr, info = env.step(np.array([hover_a] * 4, dtype=np.float32))
        if te or tr:
            break
    print(f"hover: survived {t+1}/{env.max_steps} steps, final z={env.quad.pos[2]:+.4f} "
          f"({'OK' if not te else 'terminated early!'})")

    # 3) exploration ranges + termination-cause accounting
    lo = np.full(20, np.inf); hi = np.full(20, -np.inf)
    causes = {"tilt": 0, "z": 0, "nonfinite": 0}; trunc = 0; maxrate = 0.0; maxz = 0.0
    for ep in range(n_episodes):
        obs, _ = env.reset(seed=seed0 + ep); rng = np.random.default_rng(ep)
        for t in range(env.max_steps):
            a = np.clip(hover_a + rng.normal(0, noise_std, size=4), -1, 1).astype(np.float32)
            obs, r, te, tr, info = env.step(a)
            lo = np.minimum(lo, obs); hi = np.maximum(hi, obs)
            maxrate = max(maxrate, float(np.max(np.abs(env.quad.omega))))
            maxz = max(maxz, abs(float(env.quad.pos[2])))
            if te:
                if not np.all(np.isfinite(env.quad.state)): causes["nonfinite"] += 1
                elif abs(env.quad.pos[2]) > env.max_abs_z: causes["z"] += 1
                else: causes["tilt"] += 1
                break
            if tr:
                trunc += 1; break

    print(f"\n{n_episodes} episodes: truncated={trunc} terminated_causes={causes}")
    print(f"  -> tilt terminations should be 0 (roll/pitch gate disabled). "
          f"max|z|={maxz:.2f}/{env.max_abs_z}  max|omega|={maxrate:.2f} rad/s")
    print("\nraw per-field observation ranges (target fields dropped downstream):")
    for i, n in enumerate(RAW_NAMES):
        print(f"  {n:5s} [{lo[i]:9.3f}, {hi[i]:9.3f}]")

    print("\nSuggested fixed scales (flip_env.ObsConfig): "
          "pos_scale~5, vel_scale~10, rate_scale~20, clip=10, "
          "motors via known [min_w,max_w]. Raise rate_scale if a trained flip "
          "clips its pitch-rate; a controlled 360° flip peaks well above hover exploration.")


if __name__ == "__main__":
    main()
