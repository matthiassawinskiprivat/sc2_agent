import pytest
from unittest.mock import MagicMock, AsyncMock, patch
import numpy as np
from train.envs.capture_the_flag_env import CaptureTheFlagEnv
from sc2.ids.unit_typeid import UnitTypeId
from sc2.units import Units

class RealMockUnit:
    def __init__(self, type_id, is_structure=False, owner_id=1, health=10.0):
        self.type_id = type_id
        self.is_structure = is_structure
        self.tag = 1
        self.owner_id = owner_id
        self.health = health
        self.is_memory = False

def test_termination_with_empty_army():
    """Reproduce: Trainee army is empty, should terminate."""
    with patch("train.envs.multi_client_env.MultiClientSC2Manager"):
        env = CaptureTheFlagEnv()
        env.bot1 = MagicMock()
        env.bot2 = MagicMock()
        env.bot1.is_trainee = True
        env.bot2.is_trainee = False
        
        # Pylons exist
        env.bot1.structures.return_value = MagicMock(amount=1)
        env.bot2.structures.return_value = MagicMock(amount=1)
        
        # Trainee units is empty
        env.bot1.units = Units([], env.bot1)
        env.bot1.all_units = env.bot1.units
        
        done, truncated, reward = env._check_termination()
        assert done is True, "Game should terminate when trainee units are empty"

def test_termination_with_larva_only():
    """Reproduce: Trainee has only larva, should it terminate?"""
    with patch("train.envs.multi_client_env.MultiClientSC2Manager"):
        env = CaptureTheFlagEnv()
        env.bot1 = MagicMock()
        env.bot2 = MagicMock()
        env.bot1.is_trainee = True
        env.bot2.is_trainee = False
        
        # Pylons exist
        env.bot1.structures.return_value = MagicMock(amount=1)
        env.bot2.structures.return_value = MagicMock(amount=1)
        
        # Trainee has a larva
        larva = RealMockUnit(UnitTypeId.LARVA, owner_id=1)
        env.bot1.units = Units([larva], env.bot1)
        env.bot1.all_units = env.bot1.units
        env.bot1.player_id = 1
        
        done, truncated, reward = env._check_termination()
        assert done is True, "Game should terminate if only non-combat units (like Larvae) remain"

def test_termination_with_structures_only():
    """Reproduce: Trainee has only structures, should terminate."""
    with patch("train.envs.multi_client_env.MultiClientSC2Manager"):
        env = CaptureTheFlagEnv()
        env.bot1 = MagicMock()
        env.bot2 = MagicMock()
        env.bot1.is_trainee = True
        env.bot2.is_trainee = False
        
        # Pylons exist
        env.bot1.structures.return_value = MagicMock(amount=1)
        env.bot2.structures.return_value = MagicMock(amount=1)
        
        # Trainee has only structures
        pylon = RealMockUnit(UnitTypeId.PYLON, is_structure=True, owner_id=1)
        env.bot1.units = Units([pylon], env.bot1)
        env.bot1.all_units = env.bot1.units
        env.bot1.player_id = 1
        
        done, truncated, reward = env._check_termination()
        assert done is True, "Game should terminate when only structures remain"

def test_termination_with_egg_only():
    """Reproduce: Trainee has only eggs, should terminate."""
    with patch("train.envs.multi_client_env.MultiClientSC2Manager"):
        env = CaptureTheFlagEnv()
        env.bot1 = MagicMock()
        env.bot2 = MagicMock()
        env.bot1.is_trainee = True
        env.bot2.is_trainee = False
        
        # Pylons exist
        env.bot1.structures.return_value = MagicMock(amount=1)
        env.bot2.structures.return_value = MagicMock(amount=1)
        
        # Trainee has an egg
        egg = RealMockUnit(UnitTypeId.EGG, owner_id=1)
        env.bot1.units = Units([egg], env.bot1)
        env.bot1.all_units = env.bot1.units
        env.bot1.player_id = 1
        
        done, truncated, reward = env._check_termination()
        assert done is True, "Game should terminate if only non-combat units (like Eggs) remain"

def test_continuation_with_opponent_army_lost():
    """Verify that if ONLY the opponent army is lost, the game continues."""
    with patch("train.envs.multi_client_env.MultiClientSC2Manager"):
        env = CaptureTheFlagEnv()
        env.bot1 = MagicMock()
        env.bot2 = MagicMock()
        env.bot1.is_trainee = True
        env.bot2.is_trainee = False
        
        # Pylons exist
        env.bot1.structures.return_value = MagicMock(amount=1)
        env.bot2.structures.return_value = MagicMock(amount=1)
        
        # Trainee has units
        u1 = RealMockUnit(UnitTypeId.ZERGLING, owner_id=1)
        env.bot1.units = Units([u1], env.bot1)
        env.bot1.all_units = env.bot1.units
        env.bot1.player_id = 1
        
        # Opponent has 0 units
        env.bot2.units = Units([], env.bot2)
        env.bot2.all_units = env.bot2.units
        env.bot2.player_id = 2
        
        done, truncated, reward = env._check_termination()
        assert done is False, "Game should NOT terminate when trainee still has units but opponent is dead"

@pytest.mark.asyncio
async def test_step_applies_loss_penalty():
    """Verify that the loss_penalty from config is applied in the step result."""
    with patch("train.envs.multi_client_env.MultiClientSC2Manager"):
        loss_val = -77.7
        env = CaptureTheFlagEnv(training_conf={"rewards": {"loss_penalty": loss_val}})
        env.bot1 = MagicMock()
        env.bot2 = MagicMock()
        env.bot1.is_trainee = True
        env.bot2.is_trainee = False
        
        # Mock clients and sync methods
        env.bot1.client = MagicMock()
        env.bot1.client.actions = AsyncMock()
        env.bot2.client = MagicMock()
        env.bot2.client.actions = AsyncMock()
        env._synchronized_step = AsyncMock()
        env._refresh_bot_states = AsyncMock()
        env._refresh_unit_lists = AsyncMock()
        
        # Scenario: Army is lost after step
        env.bot1.units = Units([], env.bot1) 
        env.bot1.all_units = env.bot1.units
        env.bot1.player_id = 1
        env.bot1.structures.return_value = MagicMock(amount=1)
        env.bot2.structures.return_value = MagicMock(amount=1)
        env.bot1.calculate_reward.return_value = 0.0
        env.bot1.get_observation.return_value = np.zeros((23, 128, 128))
        
        obs, reward, done, truncated, info = await env._async_step(0)
        
        assert done is True
        assert reward == loss_val

def test_termination_p1_pylon_lost():
    """Verify that if P1 pylon is lost, game ends."""
    with patch("train.envs.multi_client_env.MultiClientSC2Manager"):
        env = CaptureTheFlagEnv()
        env.bot1 = MagicMock()
        env.bot2 = MagicMock()
        env.bot1.is_trainee = True
        env.bot2.is_trainee = False
        
        # Trainee has units (so army_lost is False)
        u1 = RealMockUnit(UnitTypeId.ZERGLING, owner_id=1)
        env.bot1.units = Units([u1], env.bot1)
        env.bot1.all_units = env.bot1.units
        env.bot1.player_id = 1
        
        # P1 pylon dead, P2 pylon alive
        env.bot1.structures.return_value = MagicMock(amount=0)
        env.bot2.structures.return_value = MagicMock(amount=1)
        
        done, truncated, reward = env._check_termination()
        assert done is True
        assert reward < 0

def test_termination_p2_pylon_lost():
    """Verify that if P2 pylon is lost, game ends."""
    with patch("train.envs.multi_client_env.MultiClientSC2Manager"):
        env = CaptureTheFlagEnv()
        env.bot1 = MagicMock()
        env.bot2 = MagicMock()
        env.bot1.is_trainee = True
        env.bot2.is_trainee = False
        
        # Trainee has units (so army_lost is False)
        u1 = RealMockUnit(UnitTypeId.ZERGLING, owner_id=1)
        env.bot1.units = Units([u1], env.bot1)
        env.bot1.all_units = env.bot1.units
        env.bot1.player_id = 1
        
        # P1 pylon alive, P2 pylon dead
        env.bot1.structures.return_value = MagicMock(amount=1)
        env.bot2.structures.return_value = MagicMock(amount=0)
        
        done, truncated, reward = env._check_termination()
        assert done is True
        assert reward > 0
