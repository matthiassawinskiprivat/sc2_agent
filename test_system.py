#!/usr/bin/env python3
"""
StarCraft 2 Unit Control Agent - System Test Script
Tests all components to ensure proper functionality
"""

import sys
import os
import numpy as np
import json
import time
from pathlib import Path

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

def test_imports():
    """Test that all modules can be imported"""
    print("Testing imports...")
    
    try:
        from transformers.observation_transformer import ObservationTransformer
        print("✓ ObservationTransformer imported")
    except ImportError as e:
        print(f"✗ Failed to import ObservationTransformer: {e}")
        return False
    
    try:
        from transformers.ui_transformer import UITransformer
        print("✓ UITransformer imported")
    except ImportError as e:
        print(f"✗ Failed to import UITransformer: {e}")
        return False
    
    try:
        from agent.unit_control_agent import UnitControlAgent
        print("✓ UnitControlAgent imported")
    except ImportError as e:
        print(f"✗ Failed to import UnitControlAgent: {e}")
        return False
    
    try:
        from sc2api.unit_data_extractor import UnitDataExtractor
        print("✓ UnitDataExtractor imported")
    except ImportError as e:
        print(f"✗ Failed to import UnitDataExtractor: {e}")
        return False
    
    return True

def test_unit_data_extractor():
    """Test unit data extraction"""
    print("\nTesting unit data extraction...")
    
    try:
        extractor = UnitDataExtractor()
        unit_data = extractor.extract_unit_data()
        
        print(f"✓ Extracted data for {len(unit_data)} units")
        
        # Test some known units
        test_units = ['MARINE', 'STALKER', 'ZERGLING']
        for unit_name in test_units:
            if unit_name in unit_data:
                unit_info = unit_data[unit_name]
                print(f"  - {unit_name}: {len(unit_info.get('weapons', []))} weapons, "
                      f"health={unit_info.get('health', 0)}, "
                      f"shield={unit_info.get('shield', 0)}")
            else:
                print(f"  ⚠ {unit_name} not found in extracted data")
        
        # Save test data
        test_file = "test_unit_data.json"
        with open(test_file, 'w') as f:
            json.dump(unit_data, f, indent=2)
        print(f"✓ Saved test data to {test_file}")
        
        return True
        
    except Exception as e:
        print(f"✗ Unit data extraction failed: {e}")
        return False

def test_observation_transformer():
    """Test observation transformation"""
    print("\nTesting observation transformer...")
    
    try:
        # Create transformer with small map for testing
        transformer = ObservationTransformer(map_size=(50, 50))
        
        # Create mock observation data
        mock_observation = create_mock_sc2_observation()
        
        # Transform observation
        start_time = time.time()
        transformed = transformer.transform_observation(mock_observation, player_id=1)
        transform_time = time.time() - start_time
        
        print(f"✓ Transformation completed in {transform_time*1000:.2f}ms")
        print(f"✓ Output shape: {transformed.shape}")
        print(f"✓ Output dtype: {transformed.dtype}")
        
        # Check layer statistics
        layer_names = transformer.get_layer_names()
        for i, name in enumerate(layer_names):
            layer_data = transformed[i]
            min_val = np.min(layer_data)
            max_val = np.max(layer_data)
            mean_val = np.mean(layer_data)
            non_zero = np.count_nonzero(layer_data)
            
            print(f"  - Layer {i} ({name}): min={min_val:.3f}, max={max_val:.3f}, "
                  f"mean={mean_val:.3f}, non_zero={non_zero}")
        
        # Save test output
        np.save("test_transformed_observation.npy", transformed)
        print("✓ Saved transformed observation to test_transformed_observation.npy")
        
        return True
        
    except Exception as e:
        print(f"✗ Observation transformation failed: {e}")
        return False

def test_ui_transformer():
    """Test UI transformation"""
    print("\nTesting UI transformer...")
    
    try:
        # Create UI transformer
        ui_transformer = UITransformer(display_size=(400, 300))
        
        # Load test observation
        if os.path.exists("test_transformed_observation.npy"):
            observation = np.load("test_transformed_observation.npy")
        else:
            observation = np.random.rand(6, 50, 50)
        
        # Create composite view
        start_time = time.time()
        composite = ui_transformer.create_composite_view(observation)
        ui_time = time.time() - start_time
        
        print(f"✓ UI transformation completed in {ui_time*1000:.2f}ms")
        print(f"✓ Composite image shape: {composite.shape}")
        
        # Save test visualization
        cv2.imwrite("test_composite_view.png", composite)
        print("✓ Saved composite view to test_composite_view.png")
        
        # Test individual layer visualization
        for i in range(min(6, observation.shape[0])):
            layer_image = ui_transformer._create_single_layer_display(observation[i], i)
            filename = f"test_layer_{i}.png"
            cv2.imwrite(filename, layer_image)
            print(f"✓ Saved layer {i} visualization to {filename}")
        
        return True
        
    except Exception as e:
        print(f"✗ UI transformation failed: {e}")
        return False

def test_hierarchical_system():
    """Test hierarchical agent system"""
    print("\nTesting hierarchical agent system...")
    
    try:
        # Create hierarchical system
        from src.agent.hierarchical_agent import HierarchicalAgentSystem
        system = HierarchicalAgentSystem(map_size=(50, 50))
        
        # Create multiple agents with different layer selections
        agent1 = system.create_unit_agent("combat_agent", layer_indices=[0, 1, 2, 3])  # Attack layers only
        agent2 = system.create_unit_agent("support_agent", layer_indices=[4, 5])       # State layers only
        agent3 = system.create_unit_agent("full_agent", layer_indices=[0, 1, 2, 3, 4, 5])  # All layers
        
        print(f"✓ Created {len(system.unit_agents)} agents")
        
        # Test with mock data
        mock_observation = create_mock_sc2_observation()
        results = system.process_observation(mock_observation, player_id=1)
        
        if results:
            print(f"✓ System processed observation successfully")
            print(f"✓ {len(results)} agents returned data")
            
            for agent_id, data in results.items():
                print(f"  - Agent '{agent_id}': shape {data.shape}")
            
            # Test system statistics
            stats = system.get_system_statistics()
            print(f"✓ System statistics available: {len(stats)} metrics")
            
            # Test shared transformer
            full_stack = system.get_full_layer_stack()
            if full_stack is not None:
                print(f"✓ Full layer stack available: shape {full_stack.shape}")
            
        else:
            print("✗ System failed to process observation")
            return False
        
        return True
        
    except Exception as e:
        print(f"✗ Hierarchical system test failed: {e}")
        return False

def create_mock_sc2_observation():
    """Create mock SC2 observation for testing"""
    class MockUnit:
        def __init__(self, unit_type, x, y, owner, health, health_max, shield=0, shield_max=0, energy=0, energy_max=0):
            self.unit_type = unit_type
            self.x = x
            self.y = y
            self.owner = owner
            self.health = health
            self.health_max = health_max
            self.shield = shield
            self.shield_max = shield_max
            self.energy = energy
            self.energy_max = energy_max
    
    # Create mock units
    feature_units = [
        MockUnit(48, 50, 50, 1, 45, 45),      # Marine (allied)
        MockUnit(74, 70, 60, 1, 80, 80, 50, 50),  # Stalker (allied)
        MockUnit(105, 150, 100, 2, 35, 35),   # Zergling (enemy)
        MockUnit(110, 120, 80, 1, 125, 125),  # Marauder (allied)
        MockUnit(107, 160, 120, 2, 80, 80),   # Hydralisk (enemy)
    ]
    
    return {'feature_units': feature_units}

def cleanup_test_files():
    """Clean up test files"""
    test_files = [
        "test_unit_data.json",
        "test_transformed_observation.npy", 
        "test_composite_view.png"
    ]
    
    for i in range(6):
        test_files.append(f"test_layer_{i}.png")
    
    for file in test_files:
        if os.path.exists(file):
            os.remove(file)
            print(f"Cleaned up {file}")

def main():
    """Run all tests"""
    print("=" * 60)
    print("StarCraft 2 Unit Control Agent - System Test")
    print("=" * 60)
    
    # Clean up previous test files
    cleanup_test_files()
    
    # Run tests
    tests = [
        ("Imports", test_imports),
        ("Unit Data Extractor", test_unit_data_extractor),
        ("Observation Transformer", test_observation_transformer),
        ("UI Transformer", test_ui_transformer),
        ("Hierarchical System", test_hierarchical_system)
    ]
    
    results = []
    
    for test_name, test_func in tests:
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"✗ {test_name} failed with exception: {e}")
            results.append((test_name, False))
    
    # Summary
    print("\n" + "=" * 60)
    print("Test Results Summary")
    print("=" * 60)
    
    passed = 0
    for test_name, result in results:
        status = "✓ PASS" if result else "✗ FAIL"
        print(f"{test_name:25} {status}")
        if result:
            passed += 1
    
    print("-" * 60)
    print(f"Tests passed: {passed}/{len(results)}")
    
    if passed == len(results):
        print("\n🎉 All tests passed! System is ready for use.")
        return True
    else:
        print(f"\n⚠️  {len(results) - passed} tests failed. Check the output above for details.")
        return False


if __name__ == "__main__":
    # Add OpenCV import test
    try:
        import cv2
        print("✓ OpenCV (cv2) available")
    except ImportError:
        print("✗ OpenCV not available - UI features may not work")
        print("Install with: pip install opencv-python")
    
    # Add PyQt5 import test
    try:
        from PyQt5.QtWidgets import QApplication
        print("✓ PyQt5 available")
    except ImportError:
        print("✗ PyQt5 not available - UI will not work")
        print("Install with: pip install PyQt5")
    
    success = main()
    sys.exit(0 if success else 1)