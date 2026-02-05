import sys
import os
import glob
import re
import signal
import asyncio
import traceback
import torch
import random
import numpy as np
from PyQt5.QtWidgets import (QApplication, QMainWindow, QTabWidget, QWidget, 
                             QVBoxLayout, QLabel, QHBoxLayout, QComboBox, 
                             QPushButton, QCheckBox, QListWidget, QTableWidget, QTableWidgetItem,
                             QHeaderView, QMessageBox)
from PyQt5.QtCore import Qt, QThread, pyqtSignal
from loguru import logger

from imitation_learning.config_resolver import ConfigResolver
from imitation_learning.recorder import Recorder
from imitation_learning.replay_manager import ReplayManager
from imitation_learning.translator import ReplayTranslator
from imitation_learning.validator import Validator
from src.ui.debug_ui import DebugUI
from src.agent.maskable_recurrent_ppo import MaskableRecurrentPPO
from src.agent.maskable_ppo import MaskablePPO
from src.hierarchical_agent import HierarchicalAgentSystem
from src.ui.ui_manager import UIManager
from sc2.bot_ai import BotAI
from sc2.main import run_game
from sc2.position import Point2
from sc2 import maps
from sc2.player import Bot, Computer
from sc2.data import Difficulty, Race

class GameThread(QThread):
    finished = pyqtSignal(str)
    error = pyqtSignal(str)

    def __init__(self, recorder, level, map_name, disable_fog=False):
        super().__init__()
        self.recorder = recorder
        self.level = level
        self.map_name = map_name
        self.disable_fog = disable_fog

    def run(self):
        original_signal = signal.signal
        def no_op_signal(sig, handler): pass
        signal.signal = no_op_signal
        try:
            path = self.recorder.start_recording(self.level, self.map_name, disable_fog=self.disable_fog)
            self.finished.emit(path)
        except Exception as e:
            self.error.emit(str(e))
        finally:
            signal.signal = original_signal

class TranslationThread(QThread):
    finished = pyqtSignal(int)
    error = pyqtSignal(str)

    def __init__(self, translator, replay_path, level, config):
        super().__init__()
        self.translator = translator
        self.replay_path = replay_path
        self.level = level
        self.config = config

    def run(self):
        # Create a new event loop for this thread
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        
        # Signal hack: SC2Process tries to register signals which only works in the main thread
        original_signal = signal.signal
        def no_op_signal(sig, handler): pass
        signal.signal = no_op_signal
        
        try:
            count = self.translator.translate(self.replay_path, self.level, self.config)
            self.finished.emit(count)
        except Exception as e:
            traceback.print_exc()
            self.error.emit(str(e))
        finally:
            signal.signal = original_signal
            loop.close()

class BatchTranslationThread(QThread):
    finished = pyqtSignal(int, int) # total_replays, total_samples
    error = pyqtSignal(str)
    progress = pyqtSignal(int, int, str) # current, total, name

    def __init__(self, translator, replays, level, config):
        super().__init__()
        self.translator = translator
        self.replays = replays
        self.level = level
        self.config = config

    def run(self):
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        
        original_signal = signal.signal
        def no_op_signal(sig, handler): pass
        signal.signal = no_op_signal
        
        total_samples = 0
        try:
            for i, replay in enumerate(self.replays):
                self.progress.emit(i + 1, len(self.replays), replay["name"])
                count = self.translator.translate(replay["path"], self.level, self.config)
                total_samples += count
            self.finished.emit(len(self.replays), total_samples)
        except Exception as e:
            traceback.print_exc()
            self.error.emit(str(e))
        finally:
            signal.signal = original_signal
            loop.close()

class ValidationThread(QThread):
    finished = pyqtSignal()
    error = pyqtSignal(str)
    ui_ready = pyqtSignal(object, object) # bot, ui_manager

    def __init__(self, validator, dataset_path, level, map_name, config):
        super().__init__()
        self.validator = validator
        self.dataset_path = dataset_path
        self.level = level
        self.map_name = map_name
        self.config = config

    def run(self):
        # Create a new event loop for this thread
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        
        original_signal = signal.signal
        def no_op_signal(sig, handler): pass
        signal.signal = no_op_signal
        try:
            bot, map_path, agent_race, enemy_race, ui_manager = self.validator.start_validation(
                self.dataset_path, self.level, self.map_name, self.config
            )
            
            self.ui_ready.emit(bot, ui_manager)
            
            run_game(
                maps.Map(map_path),
                [
                    Bot(agent_race, bot),
                    Computer(enemy_race, Difficulty.Easy)
                ],
                realtime=True
            )
            self.finished.emit()
        except Exception as e:
            traceback.print_exc()
            self.error.emit(str(e))
        finally:
            signal.signal = original_signal
            loop.close()

class InferenceThread(QThread):
    finished = pyqtSignal()
    error = pyqtSignal(str)
    ui_ready = pyqtSignal(object, object) # bot, ui_manager

    def __init__(self, level, map_name, config, disable_fog=False):
        super().__init__()
        self.level = level
        self.map_name = map_name
        self.config = config
        self.disable_fog = disable_fog

    def run(self):
        # Signal hack
        original_signal = signal.signal
        signal.signal = lambda *args: None
        
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        
        try:
            # Load Latest Model
            model_dir = os.path.join("models", self.level)
            final_path = os.path.join(model_dir, "final_model.zip")
            
            if os.path.exists(final_path):
                model_path = final_path
                logger.info(f"Using final model: {model_path}")
            else:
                checkpoints = glob.glob(os.path.join(model_dir, "zerg_*_steps.zip"))
                if not checkpoints: raise FileNotFoundError(f"No model found in {model_dir}")
                
                def get_steps(path):
                    match = re.search(r"zerg_(\d+)_steps", path)
                    return int(match.group(1)) if match else 0
                
                model_path = max(checkpoints, key=get_steps)
                logger.info(f"Using highest-step checkpoint: {model_path} ({get_steps(model_path)} steps)")

            # Determine correct algorithm
            model_conf = self.config.get("model", {})
            policy_name = model_conf.get("policy", "")
            
            # Default logic: Use recurrent if explicitly requested or if policy name implies it
            is_recurrent = "Recurrent" in model_conf.get("algorithm", "") or "Lstm" in policy_name
            
            # Use specified algorithm or default based on recurrence
            algo_name = model_conf.get("algorithm")
            if not algo_name:
                algo_name = "MaskableRecurrentPPO" if is_recurrent else "MaskablePPO"
            
            AlgoClass = MaskableRecurrentPPO if "Recurrent" in algo_name else MaskablePPO
            logger.info(f"Using algorithm class: {AlgoClass.__name__} for policy {policy_name}")
            
            custom_objects = {
                "level_name": self.level,
                "imitation_data_percentage": 0.0, # Disable loading expert data for watching
                "imitation_data_max_steps": self.config["training"].get("imitation_data_max_steps", 0),
                "imitation_coef": self.config["training"].get("imitation_coef", 0.5),
                "verbose": 1
            }
            
            try:
                model = AlgoClass.load(model_path, device="cpu", custom_objects=custom_objects)
            except Exception as e:
                # Fallback: if we tried recurrent and failed, try standard MaskablePPO
                if AlgoClass == MaskableRecurrentPPO:
                    logger.warning(f"Recurrent load failed, trying fallback to MaskablePPO: {e}")
                    model = MaskablePPO.load(model_path, device="cpu", custom_objects=custom_objects)
                else:
                    raise e

            # FIX: If model expects uint8, SB3 will divide by 255.0 during predict().
            # But our features are already in float range (0-20). 
            # We force the space to float32 to bypass auto-normalization.
            from gymnasium import spaces
            if model.observation_space.dtype == np.uint8:
                logger.info(f"Correcting model observation space from uint8 to float32 to preserve signal.")
                model.observation_space = spaces.Box(
                    low=0.0, high=255.0, 
                    shape=model.observation_space.shape, 
                    dtype=np.float32
                )

            if hasattr(model, "_setup_imitation"):
                model._setup_imitation(level_name=self.level, imitation_data_percentage=0.0)
            
            # FORCE normalize_images to False. 
            # Our features are float32 raw values (0-20). 
            # If this is True, SB3 will divide by 255 during predict.
            if hasattr(model.policy, "normalize_images"):
                logger.info(f"Setting policy.normalize_images = False (was {model.policy.normalize_images})")
                model.policy.normalize_images = False

            bot = WatchBot(model, self.config, self.level, disable_fog=self.disable_fog)
            self.ui_ready.emit(bot, bot.ui_manager)
            
            from imitation_learning.recorder import Recorder
            rec = Recorder(".")
            map_path = rec.get_map_path(self.level, self.map_name)
            agent_race, enemy_race = rec._parse_races_from_map_name(self.map_name)
            
            run_game(
                maps.Map(map_path),
                [Bot(agent_race, bot), Computer(enemy_race, Difficulty.Easy)],
                realtime=False
            )
            self.finished.emit()
        except Exception as e:
            traceback.print_exc()
            self.error.emit(str(e))
        finally:
            signal.signal = original_signal
            loop.close()

from src.utils.config_manager import ConfigManager

class WatchBot(BotAI):
    def __init__(self, model, config, level_name, disable_fog=False):
        super().__init__()
        self.model = model
        self.config = config
        self.level_name = level_name
        self.disable_fog = disable_fog
        self.agent_system = None
        self.unit_command_uses_self_do = True
        
        # FIX: Enforce level_0 compatibility.
        # Even if the cumulative config says otherwise, level_0 models 
        # are strictly trained on 32x32 (1024 actions).
        if self.level_name == "level_0":
            self.grid_size = 32
        else:
            self.grid_size = self.config.get("env", {}).get("grid_size", 128)
            
        # SHARED LSTM state (matches training sequence Unit1, Unit2, Unit1, Unit2...)
        self.lstm_states = None
        
        self.ui_manager = UIManager(self)
        self.last_observation = None
        self.transformed_observation = None
        self.last_action = None
        self.command_center_tag = None
        self._objective_pos = None
        
        # Round-Robin tracking
        self.active_unit_tags = []
        self.current_unit_index = 0
        
        from sc2.ids.unit_typeid import UnitTypeId
        self.TRANSPORT_TYPES = {
            UnitTypeId.MEDIVAC, UnitTypeId.OVERLORD, 
            UnitTypeId.OVERLORDTRANSPORT, UnitTypeId.WARPPRISM,
            UnitTypeId.WARPPRISMPHASING
        }
        self.INFANTRY_TYPES = {
            UnitTypeId.MARINE, UnitTypeId.MARAUDER,
            UnitTypeId.ZERGLING, UnitTypeId.ROACH, UnitTypeId.HYDRALISK,
            UnitTypeId.ZEALOT, UnitTypeId.STALKER, UnitTypeId.ADEPT
        }

    def reset(self):
        self.lstm_states = None
        self.active_unit_tags = []
        self.current_unit_index = 0
        if self.agent_system:
            self.agent_system.current_observation_id = 0

    def action_masks(self, unit):
        agent = self.agent_system.unit_agents["agent_0"]
        # Use updated config from on_start
        action_heads = self.full_conf.get("action_heads", ["move"])
        
        masks = []
        for head in action_heads:
            if head == "move":
                masks.append(agent.get_action_mask())
            elif head == "attack":
                # Reuse move mask for attack spatial for now
                masks.append(agent.get_action_mask()) 
            elif head == "load":
                is_transport = unit.type_id in self.TRANSPORT_TYPES
                masks.append(agent.get_transport_mask(
                    self.units, 
                    self.game_info, 
                    is_transport=is_transport,
                    active_unit=unit
                ))
            elif head == "unload":
                # If unload is configured as spatial
                masks.append(agent.get_unload_mask(self.units, self.game_info))
                
        # Categorical
        cat_heads = self.full_conf.get("categorical_heads", [])
        if cat_heads:
            # Simple assumption: all categorical actions are valid
            masks.append(np.ones(len(cat_heads), dtype=bool))
            
        return np.concatenate(masks) if masks else np.array([], dtype=bool)

    def _find_objective(self):
        from sc2.ids.unit_typeid import UnitTypeId
        main_s = self.enemy_structures.filter(lambda s: s.type_id in {
            UnitTypeId.COMMANDCENTER, UnitTypeId.HATCHERY, UnitTypeId.NEXUS,
            UnitTypeId.ORBITALCOMMAND, UnitTypeId.PLANETARYFORTRESS, UnitTypeId.LAIR, UnitTypeId.HIVE,
            UnitTypeId.COMMANDCENTERFLYING, UnitTypeId.ORBITALCOMMANDFLYING
        })
        if main_s:
            t = main_s.first
            self.command_center_tag, self._objective_pos = t.tag, t.position
            logger.info(f"Objective identified: {t.type_id} at {t.position}")
            return True
        
        # Look for Beacons
        beacons = self.enemy_units.filter(lambda u: "BEACON" in u.type_id.name.upper())
        if not beacons:
            # Check allied/neutral for Level 0
            beacons = self.all_units.filter(lambda u: "BEACON" in u.type_id.name.upper())
        
        if beacons:
            t = beacons.first
            self.command_center_tag, self._objective_pos = t.tag, t.position
            logger.info(f"Objective identified (Beacon): {t.type_id} at {t.position}")
            return True
            
        return False

    async def on_start(self):
        self.reset()
        if self.disable_fog:
            await self.client.debug_show_map()

        # REPLICATE LEVEL0ENV LOGIC EXACTLY
        # Load Additive Configuration using ConfigManager
        config_manager = ConfigManager("config")
        
        # 1. Base Config Flattening
        self.full_conf = self.config.copy()
        for section in ["training", "env"]:
            if section in self.config:
                for k, v in self.config[section].items():
                    self.full_conf[k] = v
        
        # 2. Additive Properties Override
        # This ensures we get the exact same layer order as training
        active_layers = config_manager.get_cumulative_active_layers(self.level_name)
        action_heads = config_manager.get_cumulative_action_heads(self.level_name)
        categorical_heads = config_manager.get_cumulative_categorical_heads(self.level_name)
        
        self.full_conf['active_layers'] = active_layers
        self.full_conf['action_heads'] = action_heads
        self.full_conf['categorical_heads'] = categorical_heads
        
        logger.info(f"WatchBot Configured for {self.level_name}:")
        logger.info(f"  Layers ({len(active_layers)}): {active_layers}")
        logger.info(f"  Action Heads: {action_heads}")

        self.agent_system = HierarchicalAgentSystem(
            map_size=(128, 128),
            grid_size=self.grid_size,
            training_conf=self.full_conf
        )
        self.agent_system.create_unit_agent("agent_0")
        self.ui_manager.agent = self.agent_system.unit_agents["agent_0"]
        
        if active_layers:
            self.ui_manager.active_layers = active_layers

    async def _setup_level(self):
        if self.level_name == "level_0":
            await self._setup_level_0()

    async def _setup_level_0(self):
        from sc2.ids.unit_typeid import UnitTypeId
        logger.info("Setting up Level 0: Spawning Zergling and Beacon...")
        
        # 1. Clear Map
        if self.all_units:
            await self.client.debug_kill_unit([u.tag for u in self.all_units])
            await self.client.step()
        
        # 2. Determine spawn area
        area = self.game_info.playable_area
        margin = 5.0
        
        # Find valid pathable points
        grid = self.game_info.pathing_grid
        valid_points = []
        
        # Sample points (step 2 to reduce computation)
        for x in range(int(area.x + margin), int(area.x + area.width - margin), 2):
            for y in range(int(area.y + margin), int(area.y + area.height - margin), 2):
                if grid[(x, y)] != 0:
                    valid_points.append(Point2((x, y)))

        if len(valid_points) < 2:
            p1 = Point2((area.x + area.width/2, area.y + area.height/2))
            p2 = Point2((area.x + area.width/2 + 5, area.y + area.height/2))
        else:
            p1, p2 = random.sample(valid_points, 2)
        
        # Ensure minimum distance
        attempts = 0
        while p1.distance_to(p2) < 10.0 and attempts < 20:
            if len(valid_points) >= 2:
                p1, p2 = random.sample(valid_points, 2)
            attempts += 1

        # 3. Spawn units
        # Note: We assume single player watch mode where we own the units
        await self.client.debug_create_unit([[UnitTypeId.ZERGLING, 1, p1, self.player_id]])
        await self.client.debug_create_unit([[UnitTypeId.BEACON_TERRANSMALL, 1, p2, self.player_id]])
        
        # Step to materialize
        await self.client.step()
        logger.info(f"Spawned Zergling at {p1} and Beacon at {p2}")

    async def on_step(self, iteration: int):
        try:
            if iteration == 0:
                with open("debug_watchbot.log", "w") as f:
                    f.write("WatchBot Started\n")
                await self._setup_level()
                
                # Refresh state after setup so iteration 0 prediction is meaningful
                obs_data = await self.client.observation()
                from sc2.game_state import GameState
                from src.utils.proto_utils import ProtoWrapper
                game_state = GameState(obs_data.observation)
                self._prepare_step(game_state, ProtoWrapper(self.game_info._proto))
            
            # 1. Identify units we control
            allies = self.units.filter(lambda u: u.type_id in self.TRANSPORT_TYPES or u.type_id in self.INFANTRY_TYPES)
            if not allies: 
                return

            # 2. Objective tracking
            if self.command_center_tag:
                if not any(tag == self.command_center_tag for tag in [u.tag for u in self.all_units]):
                    self.command_center_tag = None
                    self._objective_pos = None

            if not self.command_center_tag:
                self._find_objective()

            target_pos = None
            if self.command_center_tag:
                target = self.all_units.find_by_tag(self.command_center_tag)
                target_pos = target.position if target else self._objective_pos

            # 3. Stable Round-Robin logic (Issue ONE command per on_step call)
            all_tags = sorted([u.tag for u in allies])
            if set(all_tags) != set(self.active_unit_tags):
                old_tag = self.active_unit_tags[self.current_unit_index] if self.active_unit_tags and self.current_unit_index < len(self.active_unit_tags) else None
                self.active_unit_tags = all_tags
                if old_tag in all_tags:
                    self.current_unit_index = all_tags.index(old_tag)
                else:
                    self.current_unit_index %= len(all_tags) if all_tags else 1
                
            if not self.active_unit_tags: return
            
            # Find the next valid unit in the cycle
            unit = None
            start_search_idx = self.current_unit_index
            while True:
                self.current_unit_index %= len(self.active_unit_tags)
                tag = self.active_unit_tags[self.current_unit_index]
                unit = self.units.find_by_tag(tag)
                if unit: break
                
                self.current_unit_index = (self.current_unit_index + 1) % len(self.active_unit_tags)
                if self.current_unit_index == start_search_idx: break
            
            if not unit: return

            # 4. Observation & Prediction
            # Treat beacon as enemy for observation (matches training)
            # Robust Beacon Search: check all entities
            beacons = self.all_units.filter(lambda u: "BEACON" in u.type_id.name.upper())
            obs_enemies = list(self.enemy_units) + list(beacons)

            # Filter allied units to NOT include the beacon
            clean_allied_units = self.units
            if beacons:
                beacon_tags = {b.tag for b in beacons}
                clean_allied_units = self.units.filter(lambda u: u.tag not in beacon_tags)

            raw_obs = {
                'units': clean_allied_units,
                'enemy_units': obs_enemies,
                'effects': self.state.effects,
                'game_info': self.game_info,
                'game_data': self.game_data,
                'creep': self.state.creep,
                'command_target': target_pos
            }
            # ONLY use precise selection for advanced levels (matches training split)
            if self.level_name != "level_0":
                raw_obs['selected_units'] = [unit]
                # Advanced levels use full context
                raw_obs['enemy_structures'] = self.enemy_structures
                raw_obs['structures'] = self.structures
            processed_obs = self.agent_system.process_observation(observation=raw_obs)["agent_0"]
            self.transformed_observation = processed_obs
            self.last_observation = processed_obs
            
            mask = self.action_masks(unit)
            is_first_step = (iteration == 0 and unit.tag == self.active_unit_tags[0])
            
            # --- Deep Probability Probing ---
            with torch.no_grad():
                obs_tensor, _ = self.model.policy.obs_to_tensor(processed_obs)
                if hasattr(self.model.policy, "get_distribution"):
                    mask_tensor = torch.as_tensor(mask).bool().to(self.model.policy.device)
                    try:
                        dist = self.model.policy.get_distribution(obs_tensor, action_masks=mask_tensor)
                        probs = dist.distribution.probs[0].detach().cpu().numpy()
                    except:
                        dist = self.model.policy.get_distribution(obs_tensor)
                        probs = dist.distribution.probs[0].detach().cpu().numpy()
                else:
                    probs = np.zeros(mask.size)

            action, self.lstm_states = self.model.predict(
                processed_obs, 
                state=self.lstm_states, 
                episode_start=np.array([is_first_step]),
                deterministic=True,
                action_masks=mask
            )
            
            # --- Ablation & Shift Testing ---
            with torch.no_grad():
                # Test 1: Zero out all units
                obs_no_units = processed_obs.copy()
                obs_no_units[0] = 0 # allied density
                obs_no_units[1] = 0 # enemy density
                action_no_units, _ = self.model.predict(obs_no_units, deterministic=True, action_masks=mask)
                
                # Test 2: Zero out objective
                obs_no_obj = processed_obs.copy()
                obs_no_obj[3] = 0 # command_objective
                action_no_obj, _ = self.model.predict(obs_no_obj, deterministic=True, action_masks=mask)

                # Test 3: Shift Test (Rotate pixels)
                obs_shifted = np.roll(processed_obs, 10, axis=2)
                action_shifted, _ = self.model.predict(obs_shifted, deterministic=True, action_masks=mask)

            # Debug Stats
            with open("debug_watchbot.log", "a") as f:
                if iteration == 0:
                    f.write(f"Model Class: {type(self.model).__name__}\n")
                    f.write(f"Model Action Space: {self.model.action_space}\n")
                    f.write(f"Model Obs Space: {self.model.observation_space}\n")
                    f.write(f"Model Obs Dtype: {self.model.observation_space.dtype}\n")
                    
                    # Probe CNN Weights
                    try:
                        ext = self.model.policy.features_extractor
                        # Check if it's UnitControlNetwork
                        f.write(f"Extractor Class: {type(ext).__name__}\n")
                        # CNN weights
                        w = ext.cnn[0].weight
                        f.write(f"CNN Layer 0 Weight: Mean={w.mean().item():.6f}, Std={w.std().item():.6f}\n")
                        if torch.isnan(w).any():
                            f.write("CRITICAL: NaNs found in model weights!\n")
                    except Exception as e:
                        f.write(f"Weight probe failed: {e}\n")

                top_idx = np.argsort(probs)[-5:][::-1]
                top_probs = probs[top_idx]
                f.write(f"Iter {iteration}: Unit: {unit.tag} Pos: {unit.position}, Action: {action}\n")
                f.write(f"  - Probs at {top_idx}: {top_probs}\n")
                f.write(f"  - Ablation: NoUnits={action_no_units}, NoObj={action_no_obj}, Shifted={action_shifted}\n")
                f.write(f"  - Mask True: {np.sum(mask)}/{mask.size}, Obs Max: {np.max(processed_obs):.2f}\n")
                f.write(f"  - Target Pos: {target_pos}\n")

            # 5. Apply & Advance index
            self._log_action(unit, action)
            self.last_action = action
            await self.apply_action(unit, action)
            self.current_unit_index = (self.current_unit_index + 1) % len(self.active_unit_tags)
            
            # 6. Watchable delay (matches training cadence in non-realtime)
            await asyncio.sleep(0.02)
            
        except Exception as e:
            logger.error(f"Error in WatchBot.on_step: {e}")
            traceback.print_exc()

    def _log_action(self, unit, action):
        grid_area = self.agent_system.grid_size * self.agent_system.grid_size
        action_heads = self.config.get("env", {}).get("action_heads", ["move"])
        total_spatial = len(action_heads) * grid_area
        
        if action < total_spatial:
            head_idx = action // grid_area
            head_name = action_heads[head_idx]
            spatial_idx = action % grid_area
            wx, wy = self.agent_system.unit_agents["agent_0"].convert_spatial_action_to_world(spatial_idx)
            # logger.debug(f"Unit {unit.tag} ({unit.type_id.name}) -> {head_name} at ({wx:.1f}, {wy:.1f})")
        else:
            cat_idx = action - total_spatial
            cat_heads = self.config.get("env", {}).get("categorical_heads", [])
            head_name = cat_heads[cat_idx] if cat_idx < len(cat_heads) else "unknown"
            # logger.debug(f"Unit {unit.tag} ({unit.type_id.name}) -> {head_name} (Categorical)")

    async def apply_action(self, unit, action):
        grid_area = self.agent_system.grid_size * self.agent_system.grid_size
        action_heads = self.config.get("env", {}).get("action_heads", ["move"])
        total_spatial = len(action_heads) * grid_area
        
        from sc2.ids.ability_id import AbilityId
        from sc2.ids.unit_typeid import UnitTypeId
        if action < total_spatial:
            head_idx = action // grid_area
            spatial_idx = action % grid_area
            head_name = action_heads[head_idx]
            
            world_x, world_y = self.agent_system.unit_agents["agent_0"].convert_spatial_action_to_world(spatial_idx)
            target = Point2((world_x, world_y))
            
            # logger.info(f"Applying Action: {head_name} to {target} (Spatial: {spatial_idx})")

            if head_name == "move": 
                self.do(unit.move(target))
            elif head_name == "attack": 
                self.do(unit.attack(target))

            elif head_name == "load":
                transport_load_mapping = {
                    UnitTypeId.MEDIVAC: AbilityId.LOAD_MEDIVAC,
                    UnitTypeId.OVERLORD: AbilityId.LOAD_OVERLORD,
                    UnitTypeId.WARPPRISM: AbilityId.LOAD_WARPPRISM,
                    UnitTypeId.WARPPRISMPHASING: AbilityId.LOAD_WARPPRISM,
                }
                candidates = self.units.filter(lambda u: u.distance_to(target) < 3.0)
                if unit.type_id in transport_load_mapping:
                    loadable = next((u for u in candidates if u.cargo_max == 0), None)
                    if loadable:
                        ability = transport_load_mapping[unit.type_id]
                        self.do(unit(ability, loadable))
                    else: self.do(unit.move(target))
                else:
                    transport = next((u for u in candidates if u.cargo_max > 0), None)
                    if transport: self.do(unit.smart(transport))
                    else: self.do(unit.move(target))
        else:
            cat_idx = action - total_spatial
            cat_heads = self.config.get("env", {}).get("categorical_heads", [])
            action_name = cat_heads[cat_idx]
            if action_name == "unload":
                mapping = {
                    UnitTypeId.MEDIVAC: AbilityId.UNLOADALLAT_MEDIVAC,
                    UnitTypeId.OVERLORD: AbilityId.UNLOADALLAT_OVERLORD,
                    UnitTypeId.OVERLORDTRANSPORT: AbilityId.UNLOADALLAT_OVERLORD,
                    UnitTypeId.WARPPRISM: AbilityId.UNLOADALLAT_WARPPRISM,
                    UnitTypeId.WARPPRISMPHASING: AbilityId.UNLOADALL_WARPPRISM,
                }
                ability = mapping.get(unit.type_id, AbilityId.UNLOADALL)
                if ability in {AbilityId.UNLOADALLAT_MEDIVAC, AbilityId.UNLOADALLAT_OVERLORD, AbilityId.UNLOADALLAT_WARPPRISM}:
                    self.do(unit(ability, unit.position))
                else:
                    self.do(unit(ability))

class RecordTab(QWidget):
    recording_finished = pyqtSignal()

    def __init__(self, config_resolver: ConfigResolver, recorder: Recorder):
        super().__init__()
        self.resolver = config_resolver
        self.recorder = recorder
        self.debug_window = None
        layout = QVBoxLayout(self)
        layout.addWidget(QLabel("Record gameplay or watch the latest trained model."))
        
        self.level_selector = QComboBox()
        self.map_selector = QComboBox()
        self.fog_checkbox = QCheckBox("Disable Fog of War")
        self.fog_checkbox.setChecked(False)
        
        self.start_button = QPushButton("Start Recording")
        self.start_button.setEnabled(False)
        self.start_button.clicked.connect(self.on_start_clicked)

        self.watch_button = QPushButton("Watch Latest Model (Deterministic)")
        self.watch_button.setEnabled(False)
        self.watch_button.clicked.connect(self.on_watch_clicked)
        
        self.level_selector.addItems(self.resolver.list_levels())
        self.level_selector.currentTextChanged.connect(self.on_level_changed)
        
        form_layout = QHBoxLayout()
        form_layout.addWidget(QLabel("Level:"))
        form_layout.addWidget(self.level_selector)
        form_layout.addWidget(QLabel("Map:"))
        form_layout.addWidget(self.map_selector)
        form_layout.addWidget(self.fog_checkbox)
        
        layout.addLayout(form_layout)
        layout.addWidget(self.start_button)
        layout.addWidget(self.watch_button)
        layout.addStretch()
        
        # Initial trigger
        if self.level_selector.count() > 0:
            self.on_level_changed(self.level_selector.currentText())

    def on_level_changed(self, level_name):
        self.map_selector.clear()
        maps = self.resolver.list_maps(level_name)
        self.map_selector.addItems(maps)
        self.start_button.setEnabled(len(maps) > 0)
        self.watch_button.setEnabled(len(maps) > 0)

    def on_start_clicked(self):
        level = self.level_selector.currentText()
        map_name = self.map_selector.currentText()
        disable_fog = self.fog_checkbox.isChecked()
        
        self.start_button.setEnabled(False)
        self.watch_button.setEnabled(False)
        self.start_button.setText("Recording in progress...")
        
        self.thread = GameThread(self.recorder, level, map_name, disable_fog=disable_fog)
        self.thread.finished.connect(self.on_recording_finished)
        self.thread.error.connect(self.on_recording_error)
        self.thread.start()

    def on_watch_clicked(self):
        level = self.level_selector.currentText()
        map_name = self.map_selector.currentText()
        config = self.resolver.get_cumulative_config(level)
        disable_fog = self.fog_checkbox.isChecked()
        
        self.start_button.setEnabled(False)
        self.watch_button.setEnabled(False)
        self.start_button.setText("Inference running...")
        
        self.thread = InferenceThread(level, map_name, config, disable_fog=disable_fog)
        self.thread.ui_ready.connect(self.show_debug_ui)
        self.thread.finished.connect(self.on_watch_finished)
        self.thread.error.connect(self.on_watch_error)
        self.thread.start()

    def show_debug_ui(self, bot, ui_manager):
        self.debug_window = DebugUI(bot, ui_manager) # bot itself serves as mock_agent
        self.debug_window.setWindowTitle(f"Model Inference: {self.level_selector.currentText()}")
        self.debug_window.show()

    def on_watch_finished(self):
        self.start_button.setEnabled(True)
        self.watch_button.setEnabled(True)
        self.watch_button.setText("Watch Latest Model (Deterministic)")
        if self.debug_window: self.debug_window.close()

    def on_watch_error(self, err):
        self.start_button.setEnabled(True)
        self.watch_button.setEnabled(True)
        self.watch_button.setText("Watch Latest Model (Deterministic)")
        QMessageBox.critical(self, "Error", f"Failed to start inference:\n{err}")
        if self.debug_window: self.debug_window.close()

    def on_recording_finished(self, path):
        self.start_button.setEnabled(True)
        self.watch_button.setEnabled(True)
        self.start_button.setText("Start Recording")
        QMessageBox.information(self, "Success", f"Replay saved to:\n{path}")
        self.recording_finished.emit()

    def on_recording_error(self, err):
        self.start_button.setEnabled(True)
        self.watch_button.setEnabled(True)
        self.start_button.setText("Start Recording")
        QMessageBox.critical(self, "Error", f"Failed to start game:\n{err}")

class TranslateTab(QWidget):
    translation_finished = pyqtSignal()

    def __init__(self, config_resolver: ConfigResolver, replay_manager: ReplayManager, translator: ReplayTranslator):
        super().__init__()
        self.resolver = config_resolver
        self.replay_manager = replay_manager
        self.translator = translator
        layout = QVBoxLayout(self)
        layout.addWidget(QLabel("Translate raw replays into training data."))
        
        # Level selection for filtering replays
        self.level_selector = QComboBox()
        self.level_selector.addItems(self.resolver.list_levels())
        self.level_selector.currentTextChanged.connect(self.refresh_list)
        
        filter_layout = QHBoxLayout()
        filter_layout.addWidget(QLabel("Show Replays for Level:"))
        filter_layout.addWidget(self.level_selector)
        layout.addLayout(filter_layout)

        self.replay_table = QTableWidget(0, 3)
        self.replay_table.setHorizontalHeaderLabels(["Name", "Date", "Size"])
        self.replay_table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.replay_table.setSelectionBehavior(QTableWidget.SelectRows)
        
        self.translate_button = QPushButton("Translate Selected")
        self.translate_button.setEnabled(False)
        self.translate_button.clicked.connect(self.on_translate_clicked)
        
        self.translate_all_button = QPushButton("Translate All")
        self.translate_all_button.clicked.connect(self.on_translate_all_clicked)
        
        self.delete_button = QPushButton("Delete Replay")
        self.delete_button.setEnabled(False)
        self.delete_button.setStyleSheet("color: red;")
        self.delete_button.clicked.connect(self.on_delete_clicked)
        
        self.replay_table.itemSelectionChanged.connect(self.on_selection_changed)
        
        layout.addWidget(self.replay_table)
        
        button_layout = QHBoxLayout()
        button_layout.addWidget(self.translate_button)
        button_layout.addWidget(self.translate_all_button)
        button_layout.addWidget(self.delete_button)
        layout.addLayout(button_layout)
        
        self.refresh_list()

    def refresh_list(self):
        level = self.level_selector.currentText()
        replays = self.replay_manager.list_raw_replays(level)
        
        self.replay_table.setRowCount(len(replays))
        for i, r in enumerate(replays):
            self.replay_table.setItem(i, 0, QTableWidgetItem(r["name"]))
            # Store the path in user data
            self.replay_table.item(i, 0).setData(Qt.UserRole, r["path"])
            self.replay_table.setItem(i, 1, QTableWidgetItem(r["date"]))
            self.replay_table.setItem(i, 2, QTableWidgetItem(r["size"]))
        
        self.translate_all_button.setEnabled(len(replays) > 0)
        self.on_selection_changed()

    def on_selection_changed(self):
        has_selection = len(self.replay_table.selectedItems()) > 0
        self.translate_button.setEnabled(has_selection)
        self.delete_button.setEnabled(has_selection)

    def on_delete_clicked(self):
        selected_items = self.replay_table.selectedItems()
        if not selected_items: return
        
        row = selected_items[0].row()
        name = self.replay_table.item(row, 0).text()
        path = self.replay_table.item(row, 0).data(Qt.UserRole)
        
        reply = QMessageBox.question(self, "Confirm Delete", 
                                   f"Are you sure you want to delete the raw replay:\n{name}?",
                                   QMessageBox.Yes | QMessageBox.No, QMessageBox.No)
        
        if reply == QMessageBox.Yes:
            try:
                if os.path.exists(path):
                    os.remove(path)
                    QMessageBox.information(self, "Deleted", "Replay file deleted successfully.")
                    self.refresh_list()
                else:
                    QMessageBox.warning(self, "Error", "File not found on disk.")
            except Exception as e:
                QMessageBox.critical(self, "Error", f"Failed to delete file:\n{e}")

    def on_translate_clicked(self):
        selected_items = self.replay_table.selectedItems()
        if not selected_items: return
        
        row = selected_items[0].row()
        replay_path = self.replay_table.item(row, 0).data(Qt.UserRole)
        level = self.level_selector.currentText()
        config = self.resolver.get_cumulative_config(level)
        
        self.translate_button.setEnabled(False)
        self.translate_all_button.setEnabled(False)
        self.translate_button.setText("Translating...")
        
        self.thread = TranslationThread(self.translator, replay_path, level, config)
        self.thread.finished.connect(self.on_translation_finished)
        self.thread.error.connect(self.on_translation_error)
        self.thread.start()

    def on_translate_all_clicked(self):
        level = self.level_selector.currentText()
        replays = self.replay_manager.list_raw_replays(level)
        if not replays: return
        
        config = self.resolver.get_cumulative_config(level)
        
        self.translate_button.setEnabled(False)
        self.translate_all_button.setEnabled(False)
        self.translate_all_button.setText("Translating All...")
        
        self.thread = BatchTranslationThread(self.translator, replays, level, config)
        self.thread.progress.connect(self.on_batch_progress)
        self.thread.finished.connect(self.on_batch_finished)
        self.thread.error.connect(self.on_translation_error)
        self.thread.start()

    def on_batch_progress(self, current, total, name):
        self.translate_all_button.setText(f"Translating ({current}/{total}): {name[:20]}...")

    def on_batch_finished(self, total_replays, total_samples):
        self.translate_button.setEnabled(True)
        self.translate_all_button.setEnabled(True)
        self.translate_all_button.setText("Translate All")
        QMessageBox.information(self, "Success", 
                                f"Batch translation complete!\n"
                                f"Processed {total_replays} replays.\n"
                                f"Generated {total_samples} training examples.")
        self.translation_finished.emit()

    def on_translation_finished(self, count):
        self.translate_button.setEnabled(True)
        self.translate_all_button.setEnabled(True)
        self.translate_button.setText("Translate Selected")
        QMessageBox.information(self, "Success", f"Translation complete!\nGenerated {count} training examples.")
        self.translation_finished.emit()

    def on_translation_error(self, err):
        self.translate_button.setEnabled(True)
        self.translate_all_button.setEnabled(True)
        self.translate_button.setText("Translate Selected")
        self.translate_all_button.setText("Translate All")
        QMessageBox.critical(self, "Error", f"Translation failed:\n{err}")

class VerifyTab(QWidget):
    def __init__(self, config_resolver: ConfigResolver, replay_manager: ReplayManager, validator: Validator):
        super().__init__()
        self.resolver = config_resolver
        self.replay_manager = replay_manager
        self.validator = validator
        self.debug_window = None
        layout = QVBoxLayout(self)
        layout.addWidget(QLabel("Verify translated data via visual playback."))
        
        # Level selection for filtering
        self.level_selector = QComboBox()
        self.level_selector.addItems(self.resolver.list_levels())
        self.level_selector.currentTextChanged.connect(self.refresh_list)
        
        filter_layout = QHBoxLayout()
        filter_layout.addWidget(QLabel("Show Data for Level:"))
        filter_layout.addWidget(self.level_selector)
        layout.addLayout(filter_layout)

        self.translated_table = QTableWidget(0, 3)
        self.translated_table.setHorizontalHeaderLabels(["Name", "Date", "Size"])
        self.translated_table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.translated_table.setSelectionBehavior(QTableWidget.SelectRows)
        
        self.verify_button = QPushButton("Verify (Watch)")
        self.verify_button.setEnabled(False)
        self.verify_button.clicked.connect(self.on_verify_clicked)
        
        self.delete_button = QPushButton("Delete Data")
        self.delete_button.setEnabled(False)
        self.delete_button.setStyleSheet("color: red;")
        self.delete_button.clicked.connect(self.on_delete_clicked)
        
        self.translated_table.itemSelectionChanged.connect(self.on_selection_changed)
        
        layout.addWidget(self.translated_table)
        
        button_layout = QHBoxLayout()
        button_layout.addWidget(self.verify_button)
        button_layout.addWidget(self.delete_button)
        layout.addLayout(button_layout)
        
        self.refresh_list()

    def refresh_list(self):
        level = self.level_selector.currentText()
        replays = self.replay_manager.list_translated_replays(level)
        
        self.translated_table.setRowCount(len(replays))
        for i, r in enumerate(replays):
            self.translated_table.setItem(i, 0, QTableWidgetItem(r["name"]))
            self.translated_table.item(i, 0).setData(Qt.UserRole, r["path"])
            self.translated_table.setItem(i, 1, QTableWidgetItem(r["date"]))
            self.translated_table.setItem(i, 2, QTableWidgetItem(r["size"]))
        
        self.on_selection_changed()

    def on_selection_changed(self):
        has_selection = len(self.translated_table.selectedItems()) > 0
        self.verify_button.setEnabled(has_selection)
        self.delete_button.setEnabled(has_selection)

    def on_delete_clicked(self):
        selected_items = self.translated_table.selectedItems()
        if not selected_items: return
        
        row = selected_items[0].row()
        name = self.translated_table.item(row, 0).text()
        path = self.translated_table.item(row, 0).data(Qt.UserRole)
        
        reply = QMessageBox.question(self, "Confirm Delete", 
                                   f"Are you sure you want to delete the translated data:\n{name}?",
                                   QMessageBox.Yes | QMessageBox.No, QMessageBox.No)
        
        if reply == QMessageBox.Yes:
            try:
                if os.path.exists(path):
                    os.remove(path)
                    QMessageBox.information(self, "Deleted", "Translated data file deleted successfully.")
                    self.refresh_list()
                else:
                    QMessageBox.warning(self, "Error", "File not found on disk.")
            except Exception as e:
                QMessageBox.critical(self, "Error", f"Failed to delete file:\n{e}")

    def on_verify_clicked(self):
        selected_items = self.translated_table.selectedItems()
        if not selected_items: return
        
        row = selected_items[0].row()
        dataset_path = self.translated_table.item(row, 0).data(Qt.UserRole)
        level = self.level_selector.currentText()
        config = self.resolver.get_cumulative_config(level)
        
        maps = self.resolver.list_maps(level)
        if not maps:
            QMessageBox.warning(self, "Warning", f"No maps found for level {level}. Cannot launch validation.")
            return
        
        map_name = maps[0] 
        filename = os.path.basename(dataset_path)
        for m in maps:
            if m in filename:
                map_name = m
                break

        self.verify_button.setEnabled(False)
        self.verify_button.setText("Validation running...")
        
        # Start game in thread
        self.thread = ValidationThread(self.validator, dataset_path, level, map_name, config)
        self.thread.ui_ready.connect(self.show_debug_ui)
        self.thread.finished.connect(self.on_validation_finished)
        self.thread.error.connect(self.on_validation_error)
        self.thread.start()

    def show_debug_ui(self, bot, ui_manager):
        # This runs in the main thread
        self.debug_window = DebugUI(bot.mock_agent, ui_manager)
        self.debug_window.setWindowTitle(f"Validation Playback")
        self.debug_window.show()

    def on_validation_finished(self):
        self.verify_button.setEnabled(True)
        self.verify_button.setText("Verify (Watch)")
        if self.debug_window:
            self.debug_window.close()

    def on_validation_error(self, err):
        self.verify_button.setEnabled(True)
        self.verify_button.setText("Verify (Watch)")
        QMessageBox.critical(self, "Error", f"Validation failed:\n{err}")
        if self.debug_window:
            self.debug_window.close()

class ImitationLearningWorkstation(QMainWindow):
    def __init__(self, project_root: str = "."):
        super().__init__()
        self.resolver = ConfigResolver(project_root)
        self.recorder = Recorder(project_root)
        self.replay_manager = ReplayManager(project_root)
        self.translator = ReplayTranslator(project_root)
        self.validator = Validator(project_root)
        
        self.setWindowTitle("SC2 Imitation Learning Workstation")
        self.resize(1000, 700)
        
        self.tabs = QTabWidget()
        self.setCentralWidget(self.tabs)
        
        self.record_tab = RecordTab(self.resolver, self.recorder)
        self.translate_tab = TranslateTab(self.resolver, self.replay_manager, self.translator)
        self.verify_tab = VerifyTab(self.resolver, self.replay_manager, self.validator)
        
        # Connect signals
        self.record_tab.recording_finished.connect(self.translate_tab.refresh_list)
        self.translate_tab.translation_finished.connect(self.verify_tab.refresh_list)
        
        self.tabs.addTab(self.record_tab, "Record")
        self.tabs.addTab(self.translate_tab, "Translate")
        self.tabs.addTab(self.verify_tab, "Verify")

def main():
    app = QApplication(sys.argv)
    window = ImitationLearningWorkstation()
    window.show()
    sys.exit(app.exec_())

if __name__ == "__main__":
    main()