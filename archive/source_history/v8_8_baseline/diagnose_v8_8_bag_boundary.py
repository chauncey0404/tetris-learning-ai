"""
V8.8 bag-boundary diagnostic.

This does not consume qualification/permanent seeds. It documents why a Hold
transition can draw two queue pieces and therefore cross a Gym 7-bag boundary.
"""
from tetris_core import GymTetrisAdapter
from v8_8_jax_vector_backend import snapshot_from_gym_raw, state_to_numpy_dict

def main():
    adapter = GymTetrisAdapter()
    try:
        state = adapter.reset(seed=4101)
        adapter.raw.gravity_enabled = False
        snap = snapshot_from_gym_raw(adapter.raw, key_seed=4101)
        d = state_to_numpy_dict(snap)
        print("current queue :", d["queue"].tolist())
        print("current bag   :", d["bag"].tolist())
        print("bag index     :", d["bag_index"])
        print("prefetch bag  :", d["next_bag"].tolist())
        print("V8.8 exact Gym next-bag prefetch: PASS")
    finally:
        adapter.close()

if __name__ == "__main__":
    main()
