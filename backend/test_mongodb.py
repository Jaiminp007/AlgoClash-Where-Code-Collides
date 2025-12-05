"""
MongoDB Connection Test Script
Run this to verify MongoDB is properly configured and accessible
"""

import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent))

from database import init_db, get_db, Generation, Simulation, Algorithm
from database.repositories import GenerationRepository, SimulationRepository
from datetime import datetime
import uuid


def test_connection():
    """Test basic MongoDB connection"""
    print("\n" + "="*60)
    print("🧪 Testing MongoDB Connection")
    print("="*60)

    try:
        db = init_db()
        print("✅ MongoDB connection successful!")
        print(f"📊 Database: {db.name}")
        print(f"📦 Collections: {db.list_collection_names()}")
        return True
    except Exception as e:
        print(f"❌ MongoDB connection failed: {e}")
        return False


def test_generation_create():
    """Test creating a generation document"""
    print("\n" + "="*60)
    print("🧪 Testing Generation Creation")
    print("="*60)

    try:
        gen_repo = GenerationRepository()

        generation = Generation(
            generation_id=f"test_gen_{uuid.uuid4()}",
            selected_models=["gpt-4", "claude-3-haiku"],
            selected_stock="AAPL_data.csv",
            status="pending",
            progress=0,
            message="Test generation"
        )

        gen_repo.create(generation)
        print(f"✅ Generation created: {generation.generation_id}")

        # Retrieve it
        retrieved = gen_repo.find_by_id(generation.generation_id)
        if retrieved:
            print(f"✅ Generation retrieved successfully")
            print(f"   Models: {retrieved['selected_models']}")
            print(f"   Stock: {retrieved['selected_stock']}")
            return True
        else:
            print("❌ Failed to retrieve generation")
            return False

    except Exception as e:
        print(f"❌ Generation test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_simulation_create():
    """Test creating a simulation document"""
    print("\n" + "="*60)
    print("🧪 Testing Simulation Creation")
    print("="*60)

    try:
        sim_repo = SimulationRepository()

        simulation = Simulation(
            simulation_id=f"test_sim_{uuid.uuid4()}",
            generation_id="test_gen_123",
            stock_ticker="AAPL",
            selected_models=["gpt-4", "claude-3-haiku"],
            status="pending",
            progress=0,
            message="Test simulation"
        )

        sim_repo.create(simulation)
        print(f"✅ Simulation created: {simulation.simulation_id}")

        # Retrieve it
        retrieved = sim_repo.find_by_id(simulation.simulation_id)
        if retrieved:
            print(f"✅ Simulation retrieved successfully")
            print(f"   Ticker: {retrieved['stock_ticker']}")
            print(f"   Status: {retrieved['status']}")
            return True
        else:
            print("❌ Failed to retrieve simulation")
            return False

    except Exception as e:
        print(f"❌ Simulation test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_update_operations():
    """Test update operations"""
    print("\n" + "="*60)
    print("🧪 Testing Update Operations")
    print("="*60)

    try:
        gen_repo = GenerationRepository()

        # Create a test generation
        gen_id = f"test_update_{uuid.uuid4()}"
        generation = Generation(
            generation_id=gen_id,
            selected_models=["gpt-4"],
            selected_stock="AAPL_data.csv",
            status="pending"
        )
        gen_repo.create(generation)
        print(f"✅ Test generation created: {gen_id}")

        # Update status
        gen_repo.update_status(gen_id, "generating", 50, "Generating algorithms...")
        print("✅ Status updated")

        # Add algorithm
        gen_repo.add_algorithm(gen_id, "gpt-4", "def execute_trade(): pass")
        print("✅ Algorithm added")

        # Retrieve and verify
        retrieved = gen_repo.find_by_id(gen_id)
        if retrieved:
            assert retrieved['status'] == 'generating', "Status not updated"
            assert retrieved['progress'] == 50, "Progress not updated"
            assert 'gpt-4' in retrieved.get('algorithms', {}), "Algorithm not added"
            print("✅ All updates verified successfully")
            return True
        else:
            print("❌ Failed to retrieve updated generation")
            return False

    except Exception as e:
        print(f"❌ Update test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_cleanup():
    """Clean up test data"""
    print("\n" + "="*60)
    print("🧹 Cleaning up test data")
    print("="*60)

    try:
        db = get_db()

        # Delete test documents
        result1 = db.generations.delete_many({'generation_id': {'$regex': '^test_'}})
        result2 = db.simulations.delete_many({'simulation_id': {'$regex': '^test_'}})

        print(f"✅ Deleted {result1.deleted_count} test generations")
        print(f"✅ Deleted {result2.deleted_count} test simulations")
        return True

    except Exception as e:
        print(f"❌ Cleanup failed: {e}")
        return False


def main():
    """Run all tests"""
    print("\n")
    print("╔" + "="*58 + "╗")
    print("║" + " "*15 + "MongoDB Test Suite" + " "*25 + "║")
    print("╚" + "="*58 + "╝")

    tests = [
        ("Connection Test", test_connection),
        ("Generation Create Test", test_generation_create),
        ("Simulation Create Test", test_simulation_create),
        ("Update Operations Test", test_update_operations),
        ("Cleanup Test", test_cleanup)
    ]

    results = []
    for name, test_func in tests:
        try:
            result = test_func()
            results.append((name, result))
        except Exception as e:
            print(f"\n❌ {name} crashed: {e}")
            results.append((name, False))

    # Summary
    print("\n" + "="*60)
    print("📊 TEST SUMMARY")
    print("="*60)

    passed = sum(1 for _, result in results if result)
    total = len(results)

    for name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{status}: {name}")

    print("="*60)
    print(f"Results: {passed}/{total} tests passed")
    print("="*60)

    if passed == total:
        print("\n🎉 All tests passed! MongoDB is ready to use.")
        return 0
    else:
        print("\n⚠️ Some tests failed. Please check MongoDB configuration.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
