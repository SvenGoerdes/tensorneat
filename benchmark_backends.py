"""Benchmark generalized vs mjx backend speed on this machine."""

import jax
import jax.numpy as jnp
import time
import numpy as np

from brax import envs

BACKENDS = ["generalized", "mjx"]
ENV_NAME = "walker2d"
WARMUP_STEPS = 20
BENCH_STEPS = 200
POP_SIZE = 512  # simulate realistic vmap over population


def bench_single_env(backend: str):
    """Single env, sequential steps."""
    env = envs.create(ENV_NAME, backend=backend)
    key = jax.random.PRNGKey(0)
    action = jnp.zeros(env.action_size)

    step_fn = jax.jit(env.step)
    state = env.reset(key)

    # warmup
    for _ in range(WARMUP_STEPS):
        state = step_fn(state, action)
    state.obs.block_until_ready()

    # bench
    state = env.reset(key)
    tic = time.time()
    for _ in range(BENCH_STEPS):
        state = step_fn(state, action)
    state.obs.block_until_ready()
    elapsed = time.time() - tic

    return elapsed, elapsed / BENCH_STEPS * 1000


def bench_vmap_population(backend: str, pop_size: int):
    """vmap over population — closer to actual NEAT eval."""
    env = envs.create(ENV_NAME, backend=backend)
    keys = jax.random.split(jax.random.PRNGKey(0), pop_size)
    actions = jnp.zeros((pop_size, env.action_size))

    reset_fn  = jax.jit(jax.vmap(env.reset))
    step_fn   = jax.jit(jax.vmap(env.step))

    states = reset_fn(keys)

    # warmup
    for _ in range(WARMUP_STEPS):
        states = step_fn(states, actions)
    states.obs.block_until_ready()

    # bench
    states = reset_fn(keys)
    tic = time.time()
    for _ in range(BENCH_STEPS):
        states = step_fn(states, actions)
    states.obs.block_until_ready()
    elapsed = time.time() - tic

    return elapsed, elapsed / BENCH_STEPS * 1000


def main():
    print(f"JAX devices: {jax.devices()}")
    print(f"Env: {ENV_NAME} | Bench steps: {BENCH_STEPS} | Pop size: {POP_SIZE}")
    print("=" * 65)

    print("\n--- Single environment (sequential) ---")
    print(f"{'Backend':<15} {'Total (s)':>10} {'ms/step':>10}")
    print("-" * 40)
    results_single = {}
    for backend in BACKENDS:
        try:
            elapsed, ms_per_step = bench_single_env(backend)
            results_single[backend] = ms_per_step
            print(f"{backend:<15} {elapsed:>10.2f} {ms_per_step:>10.2f}")
        except Exception as e:
            print(f"{backend:<15} ERROR: {e}")

    if len(results_single) == 2:
        ratio = results_single["mjx"] / results_single["generalized"]
        print(f"\n  mjx is {ratio:.2f}x {'slower' if ratio > 1 else 'faster'} than generalized (single env)")

    print(f"\n--- vmap over population (pop_size={POP_SIZE}) ---")
    print(f"{'Backend':<15} {'Total (s)':>10} {'ms/step':>10} {'ms/step/ind':>14}")
    print("-" * 55)
    results_vmap = {}
    for backend in BACKENDS:
        try:
            elapsed, ms_per_step = bench_vmap_population(backend, POP_SIZE)
            ms_per_ind = ms_per_step / POP_SIZE
            results_vmap[backend] = ms_per_step
            print(f"{backend:<15} {elapsed:>10.2f} {ms_per_step:>10.2f} {ms_per_ind:>14.4f}")
        except Exception as e:
            print(f"{backend:<15} ERROR: {e}")

    if len(results_vmap) == 2:
        ratio = results_vmap["mjx"] / results_vmap["generalized"]
        print(f"\n  mjx is {ratio:.2f}x {'slower' if ratio > 1 else 'faster'} than generalized (vmap pop)")

    print("\n" + "=" * 65)
    print("Recommendation:")
    if len(results_vmap) == 2:
        ratio = results_vmap["mjx"] / results_vmap["generalized"]
        if ratio < 1.5:
            print("  -> Use mjx: speed difference is acceptable, physics is more reliable.")
        elif ratio < 3.0:
            print("  -> Consider mjx: moderate slowdown but significantly better physics.")
        else:
            print("  -> mjx is significantly slower. Weigh carefully against physics quality.")


if __name__ == "__main__":
    main()
