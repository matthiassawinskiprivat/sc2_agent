import numpy as np

class PerspectiveTransformer:
    """
    Utility for spatial transformations to support perspective flipping in self-play.
    Rotates observations and actions by 180 degrees to maintain a canonical view.
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
