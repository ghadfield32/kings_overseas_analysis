#!/usr/bin/env python3
"""
Integration tests for the modular basketball scouting pipeline.
Tests module interactions and data flow.
"""

import sys
import traceback
from pathlib import Path

def test_module_imports():
    """Test that all modules can be imported."""
    print("=" * 60)
    print("TEST 1: MODULE IMPORTS")
    print("=" * 60)

    try:
        from data import DataLoader
        print("✓ DataLoader imported")

        from database import DatabaseManager
        print("✓ DatabaseManager imported")

        from features import calculate_statistics, StatisticsCalculator
        print("✓ Statistics functions imported")

        from analysis import analyze_performance_patterns, PerformanceAnalyzer
        print("✓ Performance analysis imported")

        from models import train_nba_success_model, NBASuccessPredictor
        print("✓ ML models imported")

        from scouting import identify_scouting_targets, ScoutingAnalyzer
        print("✓ Scouting analysis imported")

        from reporting import generate_comprehensive_report, ReportGenerator
        print("✓ Reporting functions imported")

        print("\n✓ All module imports successful!")
        return True

    except ImportError as e:
        print(f"\n✗ Import error: {e}")
        return False
    except Exception as e:
        print(f"\n✗ Error: {e}")
        return False

def test_utils_functions():
    """Test utility functions work correctly."""
    print("\n" + "=" * 60)
    print("TEST 2: UTILITY FUNCTIONS")
    print("=" * 60)

    try:
        from utils import safe_div, create_player_id, calculate_age_bonus

        # Test safe_div
        assert safe_div(10, 2) == 5.0, "safe_div failed"
        print("✓ safe_div works")

        # Test create_player_id
        import pandas as pd
        pid = create_player_id(pd.Series(['John']), pd.Series(['Doe']))
        assert pid.iloc[0] == 'john_doe', "create_player_id failed"
        print("✓ create_player_id works")

        # Test age bonus
        assert calculate_age_bonus(22) == 1.3, "age bonus failed"
        print("✓ calculate_age_bonus works")

        print("\n✓ All utility functions work!")
        return True

    except Exception as e:
        print(f"\n✗ Error: {e}")
        traceback.print_exc()
        return False

def test_data_loader():
    """Test data loader with synthetic data."""
    print("\n" + "=" * 60)
    print("TEST 3: DATA LOADER")
    print("=" * 60)

    try:
        from data import DataLoader
        import tempfile
        import json
        import shutil

        # Create temp directory with test data
        temp_dir = tempfile.mkdtemp()

        test_data = [
            {"first_name": "Test", "last_name": "Player", "birth_date": "1995-01-15"}
        ]

        with open(Path(temp_dir) / 'player.json', 'w') as f:
            json.dump(test_data, f)
        with open(Path(temp_dir) / 'nba_box_player_season.json', 'w') as f:
            json.dump([], f)
        with open(Path(temp_dir) / 'international_box_player_season.json', 'w') as f:
            json.dump([], f)

        # Test loader
        loader = DataLoader(data_dir=temp_dir)
        player_df, nba_df, intl_df, issues = loader.load_and_validate()

        assert len(player_df) == 1, "Wrong player count"
        assert 'player_id' in player_df.columns, "Missing player_id"
        print("✓ DataLoader works with synthetic data")

        # Cleanup
        shutil.rmtree(temp_dir)
        print("✓ Cleanup successful")

        return True

    except Exception as e:
        print(f"\n✗ Error: {e}")
        traceback.print_exc()
        return False

def test_database_manager():
    """Test database manager with in-memory database."""
    print("\n" + "=" * 60)
    print("TEST 4: DATABASE MANAGER")
    print("=" * 60)

    try:
        from database import DatabaseManager
        import pandas as pd

        # Create test data
        player_df = pd.DataFrame({
            'player_id': ['test_player'],
            'first_name': ['Test'],
            'last_name': ['Player'],
            'birth_date': ['1995-01-15'],
            'birth_year': [1995],
            'age_2021': [26]
        })

        nba_df = pd.DataFrame({
            'player_id': ['test_player'],
            'season': [2021],
            'team': ['TestTeam'],
            'games': [50],
            'minutes': [1200.0],
            'points': [600.0],
            'assists': [200.0],
            'offensive_rebounds': [50.0],
            'defensive_rebounds': [150.0]
        })

        intl_df = pd.DataFrame()
        quality_issues = []

        # Test database
        db = DatabaseManager(db_path=':memory:')
        db.connect()
        db.create_schema()
        db.create_indexes()
        db.load_data(player_df, nba_df, intl_df, quality_issues)

        # Verify data
        query = "SELECT COUNT(*) FROM players"
        count = pd.read_sql_query(query, db.conn).iloc[0, 0]
        assert count == 1, f"Expected 1 player, got {count}"

        db.close()
        print("✓ DatabaseManager works with in-memory database")

        return True

    except Exception as e:
        print(f"\n✗ Error: {e}")
        traceback.print_exc()
        return False

def test_statistics_calculator():
    """Test statistics calculator."""
    print("\n" + "=" * 60)
    print("TEST 5: STATISTICS CALCULATOR")
    print("=" * 60)

    try:
        from features import StatisticsCalculator
        import pandas as pd

        # Create test data
        test_data = pd.DataFrame({
            'games': [50, 60, 70],
            'minutes': [1200, 1500, 1800],
            'points': [600, 900, 1200],
            'assists': [200, 300, 250],
            'offensive_rebounds': [50, 40, 60],
            'defensive_rebounds': [150, 180, 200],
            'two_points_made': [150, 200, 250],
            'two_points_attempted': [300, 400, 500],
            'three_points_made': [50, 100, 100],
            'three_points_attempted': [150, 300, 300],
            'free_throws_made': [100, 100, 100],
            'free_throws_attempted': [120, 130, 140]
        })

        # Test calculator
        calc = StatisticsCalculator()
        result = calc.calculate_all_statistics(test_data, 'Test')

        assert 'ppg' in result.columns, "Missing PPG column"
        assert 'efficiency' in result.columns, "Missing efficiency column"
        assert result['ppg'].iloc[0] == 12.0, "PPG calculation wrong"

        print("✓ StatisticsCalculator works correctly")

        return True

    except Exception as e:
        print(f"\n✗ Error: {e}")
        traceback.print_exc()
        return False

def main():
    """Run all integration tests."""
    print("BASKETBALL SCOUTING PIPELINE - INTEGRATION TESTS")
    print("=" * 60)

    tests = [
        test_module_imports,
        test_utils_functions,
        test_data_loader,
        test_database_manager,
        test_statistics_calculator
    ]

    passed = 0
    total = len(tests)

    for test in tests:
        try:
            if test():
                passed += 1
        except Exception as e:
            print(f"\n✗ Test {test.__name__} failed with exception: {e}")

    print("\n" + "=" * 60)
    print("INTEGRATION TEST RESULTS")
    print("=" * 60)
    print(f"Passed: {passed}/{total}")

    if passed == total:
        print("✓ ALL TESTS PASSED!")
        print("✓ Pipeline is ready for use")
        return True
    else:
        print("✗ SOME TESTS FAILED")
        print("✗ Check errors above and fix issues")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
