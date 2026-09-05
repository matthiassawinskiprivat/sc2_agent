"""DiskDictRolloutBuffer RNN sequence boundaries respect episode_starts."""
import numpy as np
from gymnasium import spaces

from src.agent.utils.disk_buffer import (
    DEFAULT_RNN_MAX_SEQ_LEN,
    DiskDictRolloutBuffer,
    build_recurrent_sequence_spans,
    canonical_recurrent_batch_shape,
    pack_recurrent_sequence_batches,
    padded_sequence_batch_timesteps,
)


def test_splits_on_focal_unit_episode_starts():
    episode_starts = np.zeros((16, 1), dtype=np.float32)
    episode_starts[5, 0] = 1.0
    episode_starts[10, 0] = 1.0

    spans = build_recurrent_sequence_spans(1, 16, episode_starts, max_seq_len=32)
    assert spans == [(0, 0, 5), (0, 5, 5), (0, 10, 6)]


def test_splits_on_max_seq_len_without_resets():
    episode_starts = np.zeros((64, 2), dtype=np.float32)

    spans = build_recurrent_sequence_spans(2, 64, episode_starts, max_seq_len=32)
    assert spans == [
        (0, 0, 32),
        (0, 32, 32),
        (1, 0, 32),
        (1, 32, 32),
    ]


def test_reset_at_chunk_boundary_starts_new_span():
    episode_starts = np.zeros((40, 1), dtype=np.float32)
    episode_starts[32, 0] = 1.0

    spans = build_recurrent_sequence_spans(1, 40, episode_starts, max_seq_len=32)
    assert spans == [(0, 0, 32), (0, 32, 8)]


def test_skips_inactive_env_rows_in_multi_env_buffer():
    """num_envs=2: do not run env0 RNN through timesteps where only env1 acted."""
    episode_starts = np.zeros((6, 2), dtype=np.float32)
    policy_active = np.array(
        [
            [1.0, 0.0],
            [1.0, 0.0],
            [0.0, 1.0],
            [1.0, 0.0],
            [1.0, 0.0],
            [0.0, 1.0],
        ],
        dtype=np.float32,
    )

    spans = build_recurrent_sequence_spans(
        2, 6, episode_starts, max_seq_len=32, policy_active=policy_active
    )
    assert spans == [(0, 0, 2), (0, 3, 2), (1, 2, 1), (1, 5, 1)]


def test_pack_equal_length_spans_uses_unpadded_equivalent():
    spans = [(0, 0, 32), (0, 32, 32), (1, 0, 32), (1, 32, 32)]
    batches = pack_recurrent_sequence_batches(spans, 128)
    assert batches == [spans]
    assert padded_sequence_batch_timesteps(batches[0]) == 128


def test_pack_does_not_mix_long_span_with_many_short_spans():
    """Regression: unpadded packing put one len-32 span with 96 len-1 spans.

    Pad-to-max then uploaded 97*32 map frames (~6 GB at CTF channel counts).
    """
    spans = [(0, 0, 32)] + [(0, t, 1) for t in range(1, 97)]
    batches = pack_recurrent_sequence_batches(spans, 128)
    for batch in batches:
        assert padded_sequence_batch_timesteps(batch) <= 128
    assert any(span[2] == 32 for span in spans)
    # The long span must sit in a batch that is not padded out by dozens of 1-step spans.
    long_batch = next(batch for batch in batches if any(s[2] == 32 for s in batch))
    assert len(long_batch) <= 4


def test_pack_single_span_longer_than_batch_size_is_kept():
    batches = pack_recurrent_sequence_batches([(0, 0, 32)], 16)
    assert batches == [[(0, 0, 32)]]


def test_get_padded_map_batch_stays_within_batch_size(tmp_path):
    """End-to-end: mixed episode cuts must not explode flattened map minibatches."""
    obs_space = spaces.Dict(
        {
            "map": spaces.Box(0, 1, (2, 8, 8), dtype=np.float32),
            "command": spaces.Box(0, 1, (4,), dtype=np.float32),
        }
    )
    buf = DiskDictRolloutBuffer(
        40,
        obs_space,
        spaces.MultiDiscrete([2, 4]),
        device="cpu",
        n_envs=1,
        temp_dir=str(tmp_path),
        storage_backend="ram",
        rnn_type="gru",
        hidden_size=8,
    )
    buf.full = True
    buf.pos = 0
    buf.policy_active[:] = 1.0
    buf.episode_starts[:] = 0.0
    buf.episode_starts[0, 0] = 1.0
    for t in range(8, 40):
        buf.episode_starts[t, 0] = 1.0

    batch_size = 16
    n_seq, seq_len = canonical_recurrent_batch_shape(batch_size, DEFAULT_RNN_MAX_SEQ_LEN)
    expected = n_seq * seq_len
    for sample in buf.get(batch_size):
        assert sample.observations["map"].shape[0] == expected
        assert sample.observations["map"].shape[0] <= max(batch_size, DEFAULT_RNN_MAX_SEQ_LEN)


def test_canonical_get_uses_one_map_shape_for_mixed_spans(tmp_path):
    """Variable episode cuts must not produce mixed GPU batch sizes (VRAM ratchet)."""
    obs_space = spaces.Dict(
        {
            "map": spaces.Box(0, 1, (2, 8, 8), dtype=np.float32),
            "command": spaces.Box(0, 1, (4,), dtype=np.float32),
        }
    )
    buf = DiskDictRolloutBuffer(
        64,
        obs_space,
        spaces.MultiDiscrete([2, 4]),
        device="cpu",
        n_envs=1,
        temp_dir=str(tmp_path),
        storage_backend="ram",
        rnn_type="gru",
        hidden_size=8,
    )
    buf.full = True
    buf.pos = 0
    buf.policy_active[:] = 1.0
    buf.episode_starts[:] = 0.0
    buf.episode_starts[0, 0] = 1.0
    buf.episode_starts[32, 0] = 1.0
    for t in range(40, 64):
        buf.episode_starts[t, 0] = 1.0

    batch_size = 256
    n_seq, seq_len = canonical_recurrent_batch_shape(batch_size, DEFAULT_RNN_MAX_SEQ_LEN)
    expected = n_seq * seq_len
    shapes = []
    mask_sums = []
    for sample in buf.get(batch_size):
        shapes.append(int(sample.observations["map"].shape[0]))
        mask_sums.append(float(sample.mask.sum()))
        assert shapes[-1] == expected
        assert sample.lstm_states.pi[0].shape[1] == n_seq
    assert len(set(shapes)) == 1
    assert all(s < expected for s in mask_sums)


def test_pack_caps_sequence_count_not_in_batch_max_len():
    """Five length-1 spans at batch_size=128 → two batches (cap 4), not one 5×1 batch."""
    spans = [(0, t, 1) for t in range(5)]
    batches = pack_recurrent_sequence_batches(spans, 128)
    assert [len(b) for b in batches] == [4, 1]
    assert padded_sequence_batch_timesteps(batches[0]) == 4 * DEFAULT_RNN_MAX_SEQ_LEN
