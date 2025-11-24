# StarCraft 2 Unit Control Agent

A sophisticated StarCraft 2 AI agent focused on unit control using reinforcement learning techniques. This project implements a hierarchical agent architecture with a specialized unit control agent that processes game observations into CNN-ready formats.

## Features

### Core Components
- **Observation Transformer**: Converts SC2 observations into 6-layer CNN format
- **Real-time Debug UI**: Visual interface for monitoring agent behavior
- **Unit Data Extractor**: Comprehensive unit statistics from SC2 API
- **Hierarchical Architecture**: Designed for future master agent integration

### Observation Layers
1. **Allied Ground Attack Heatmap**: Damage zones for allied ground units
2. **Allied Air Attack Heatmap**: Damage zones for allied air units  
3. **Enemy Ground Attack Heatmap**: Damage zones for enemy ground units
4. **Enemy Air Attack Heatmap**: Damage zones for enemy air units
5. **Health Percentage**: Average health per map cell
6. **Energy Percentage**: Average energy for spell casters per map cell

### Key Capabilities
- ✅ Real-time observation transformation
- ✅ Interactive debug visualization
- ✅ Layer toggle controls
- ✅ Performance monitoring
- ✅ Unit transformation awareness (Vikings, Siege Tanks, etc.)
- ✅ Comprehensive unit database
- ✅ Extensible architecture for future RL integration

## Installation

### Prerequisites
- Python 3.7 or higher
- StarCraft 2 (any version, maps required)
- Windows, macOS, or Linux

### Quick Setup
```bash
# Clone the repository
git clone <repository-url>
cd sc2_unit_agent

# Run setup script
python setup.py

# Or manually:
pip install -r requirements.txt
python main.py --setup
```

### Manual Installation
```bash
# Install PyTorch first
pip install torch torchvision

# Install other dependencies  
pip install -r requirements.txt

# Extract unit data
python main.py --setup
```

## Usage

### Demo Mode
Run with simulated data to test the system:
```bash
# With UI (recommended)
python main.py --demo

# Headless mode
python main.py --demo --no-ui

# Save debug data
python main.py --demo --save-debug
```

### Interactive Controls
- **Layer Toggles**: Enable/disable individual observation layers
- **Real-time Mode**: Toggle continuous updates
- **Reset Agent**: Clear current state and statistics
- **Save Debug**: Export current observation data

### Configuration
Edit `config.json` to customize:
```json
{
  "map_size": [200, 200],
  "display_size": [1200, 800],
  "fps_limit": 60,
  "enable_ui": true,
  "save_debug_data": false
}
```

## Architecture

### Project Structure
```
sc2_unit_agent/
├── src/
│   ├── agent/
│   │   └── unit_control_agent.py      # Main agent class
│   ├── transformers/
│   │   ├── observation_transformer.py # CNN data transformation
│   │   └── ui_transformer.py          # Visualization preparation
│   ├── sc2api/
│   │   └── unit_data_extractor.py     # Unit statistics extraction
│   └── utils/                         # Utility functions
├── ui/
│   └── debug_ui.py                    # Real-time debug interface
├── data/
│   └── unit_data.json                 # Extracted unit statistics
├── main.py                            # Entry point
├── setup.py                           # Installation script
└── config.json                        # Configuration file
```

### Data Flow
1. **SC2 Observation** → Raw game state from python-sc2
2. **Observation Transformer** → 6-layer CNN format [L×W×H]
3. **UI Transformer** → RGB visualization
4. **Debug UI** → Real-time display with controls

### Layer Processing
```
Game Observation
      ↓
Unit Position/Stats Extraction
      ↓
Attack Range → Circular Heatmap (damage/100, clipped at 1.0)
Health/Shield → Average percentage per cell
Energy → Average percentage (spell casters only)
      ↓
6-Layer Output [AlliedGround, AlliedAir, EnemyGround, EnemyAir, Health, Energy]
```

## Unit Support

### Fully Supported Units
- **Terran**: Marine, Marauder, Siege Tank, Viking, Battlecruiser, Thor
- **Protoss**: Zealot, Stalker, Immortal, Colossus, Void Ray, Carrier
- **Zerg**: Zergling, Hydralisk, Ultralisk, Mutalisk, Brood Lord

### Transformation-Aware Units
- Vikings (Assault/Fighter modes)
- Siege Tanks (Mobile/Sieged modes)  
- Thors (Explosive/High Impact modes)
- Lurkers (Burrowed/Unburrowed)

### Unit Data
Each unit includes:
- Attack damage and range (ground/air)
- Health and shield values
- Energy for spell casters
- Transformation abilities
- Movement type (ground/flying)

## Development

### Adding New Units
1. Update `unit_data_extractor.py` with new unit statistics
2. Run `python main.py --setup` to regenerate unit data
3. Test with demo mode to verify visualization

### Custom Observation Layers
1. Modify `observation_transformer.py` to add new layers
2. Update `ui_transformer.py` for visualization
3. Add UI controls in `debug_ui.py`

### Integration with RL
The observation format is designed for CNN-based RL algorithms:
- Fixed-size input tensors
- Normalized values [0,1]
- Spatial relationships preserved
- Ready for batch processing

## Performance

### System Requirements
- **Minimum**: 4GB RAM, integrated graphics
- **Recommended**: 8GB RAM, dedicated GPU
- **Optimal**: 16GB RAM, RTX 2060 or better

### Benchmarks
- Observation transformation: ~0.1ms per frame
- UI rendering: ~16ms per frame (60 FPS)
- Memory usage: ~50MB for full system

### Optimization Tips
- Reduce map_size for faster processing
- Disable unnecessary UI layers
- Use headless mode for training
- Batch observations when possible

## Troubleshooting

### Common Issues

**Import Errors**
```bash
pip install --upgrade pip
pip install -r requirements.txt --force-reinstall
```

**StarCraft 2 Not Found**
- Install StarCraft 2 from Battle.net
- Ensure maps are available
- Set sc2_path in config.json manually

**UI Not Displaying**
- Install PyQt5: `pip install PyQt5`
- Check display drivers
- Try headless mode: `--no-ui`

**Performance Issues**
- Reduce display_size in config
- Lower fps_limit
- Disable expensive UI layers

### Debug Mode
Enable debug output:
```bash
python main.py --demo --save-debug
```

This saves:
- Raw observation data (JSON)
- Transformed tensors (NPY)
- Visualizations (PNG)
- Statistics (JSON)

## Future Enhancements

### Planned Features
- **Master Agent**: Strategic decision-making layer
- **RL Integration**: Deep Q-Networks, PPO, A3C
- **Multi-agent Support**: Multiple unit control agents
- **Replay Analysis**: Process recorded games
- **Model Training**: End-to-end learning pipeline
- **SC2 AI Arena**: Competition-ready agent

### Research Directions
- Hierarchical reinforcement learning
- Multi-agent coordination
- Transfer learning between unit types
- Meta-learning for new strategies
- Real-time adaptation

## Contributing

1. Fork the repository
2. Create a feature branch
3. Add tests for new functionality
4. Submit a pull request

## License

This project is licensed under the MIT License - see LICENSE file for details.

## Acknowledgments

- **Blizzard Entertainment**: StarCraft 2 and SC2 API
- **DeepMind**: pysc2 library and research inspiration  
- **SC2 AI Community**: Maps, tools, and competition frameworks
- **PyTorch Team**: Deep learning framework

## References

- [StarCraft II API Documentation](https://github.com/Blizzard/s2client-api)
- [pysc2 - DeepMind's SC2 Learning Environment](https://github.com/deepmind/pysc2)
- [SC2 AI Arena](https://sc2aiarena.com/)
- [StarCraft 2 AI - Lecture Series](https://www.youtube.com/playlist?list=PLoKL1pk7xHAySrlU2W4kR6gUqR3f0CUKe)