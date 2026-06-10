"""DiskDictRolloutBuffer RNN sequence boundaries respect episode_starts."""
import numpy as np

from src.agent.utils.disk_buffer import build_recurrent_sequence_spans


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
