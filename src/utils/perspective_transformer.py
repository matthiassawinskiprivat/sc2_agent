import numpy as np

from src.agent.squad_constants import SQUAD_SPATIAL_RESOLUTION


class PerspectiveTransformer:
    """180° rotation helpers (legacy unit tests / optional tooling).

    CTF/squad training uses owner-id perspective only: P1 and P2 see the same
    world-coordinate maps, scalars, and spatial actions as old P1-only training.
    """

    @staticmethod
    def flip_observation(obs: np.ndarray) -> np.ndarray:
        """
        Rotates a [C, H, W] observation by 180 degrees.
        """
        # np.flip on H and W axes
        return np.flip(obs, axis=(-2, -1)).copy()

    @staticmethod
    def flip_action_idx(action_idx: int, grid_size: int) -> int:
        """
        Rotates a spatial action index by 180 degrees.
        Handles dual-head (Move/Attack) if action_idx >= grid_size * grid_size.
        """
        grid_area = grid_size * grid_size
        
        is_attack = False
        spatial_idx = action_idx
        
        if action_idx >= grid_area:
            is_attack = True
            spatial_idx = action_idx - grid_area
            
        y, x = divmod(spatial_idx, grid_size)
        
        y_new = (grid_size - 1) - y
        x_new = (grid_size - 1) - x
        
        spatial_idx_new = y_new * grid_size + x_new
        
        if is_attack:
            return spatial_idx_new + grid_area
        return spatial_idx_new

    @staticmethod
    def flip_mask(mask: np.ndarray, grid_size: int) -> np.ndarray:
        """
        Flips a flat action mask [2 * grid_size * grid_size].
        """
        grid_area = grid_size * grid_size
        
        if len(mask) == grid_area:
            # Single head
            mask_2d = mask.reshape((grid_size, grid_size))
            return np.flip(mask_2d, axis=(0, 1)).flatten()
        
        if len(mask) == 2 * grid_area:
            # Dual head
            move_mask = mask[:grid_area].reshape((grid_size, grid_size))
            attack_mask = mask[grid_area:].reshape((grid_size, grid_size))
            
            move_flipped = np.flip(move_mask, axis=(0, 1)).flatten()
            attack_flipped = np.flip(attack_mask, axis=(0, 1)).flatten()
            
            return np.concatenate([move_flipped, attack_flipped])
            
        return np.flip(mask) # Fallback

    @staticmethod
    def decode_world_spatial_idx(
        head_name: str,
        spatial_idx: int,
        owner_id: int,
        grid_size: int = SQUAD_SPATIAL_RESOLUTION,
    ) -> int:
        """Pass-through: no P2 spatial un-flip (world coords for all owners)."""
        del head_name, owner_id, grid_size
        return int(spatial_idx)
