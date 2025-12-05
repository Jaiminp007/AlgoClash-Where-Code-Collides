"""
Clear MongoDB Database Script
Deletes all data from the AlgoClash MongoDB database
"""

from database import init_db, get_db

def clear_database():
    """Clear all collections in the database"""
    print("\n" + "="*60)
    print("🗑️  CLEARING MONGODB DATABASE")
    print("="*60)

    try:
        # Initialize database
        db = init_db()

        # Get all collection names
        collections = db.list_collection_names()

        if not collections:
            print("ℹ️  Database is already empty")
            return

        print(f"\n📦 Found {len(collections)} collections:")
        for col in collections:
            print(f"   - {col}")

        # Ask for confirmation
        print("\n⚠️  WARNING: This will delete ALL data!")
        response = input("Are you sure you want to continue? (yes/no): ")

        if response.lower() not in ['yes', 'y']:
            print("❌ Operation cancelled")
            return

        # Delete all documents from each collection
        total_deleted = 0
        for collection_name in collections:
            collection = db[collection_name]
            count = collection.count_documents({})

            if count > 0:
                result = collection.delete_many({})
                print(f"✅ Deleted {result.deleted_count} documents from '{collection_name}'")
                total_deleted += result.deleted_count
            else:
                print(f"ℹ️  Collection '{collection_name}' was already empty")

        print("\n" + "="*60)
        print(f"✅ CLEANUP COMPLETE")
        print(f"   Total documents deleted: {total_deleted}")
        print("="*60)

    except Exception as e:
        print(f"\n❌ Error clearing database: {e}")
        import traceback
        traceback.print_exc()


def drop_database():
    """Completely drop the database (more aggressive)"""
    print("\n" + "="*60)
    print("💣 DROPPING ENTIRE DATABASE")
    print("="*60)

    try:
        # Initialize database
        db = init_db()
        db_name = db.name

        print(f"\n⚠️  WARNING: This will DELETE the entire '{db_name}' database!")
        response = input("Are you sure? Type 'DROP DATABASE' to confirm: ")

        if response != 'DROP DATABASE':
            print("❌ Operation cancelled")
            return

        # Drop the database
        from database.connection import _mongodb
        client = _mongodb._client
        client.drop_database(db_name)

        print(f"✅ Database '{db_name}' has been dropped")
        print("ℹ️  It will be recreated on next connection")

    except Exception as e:
        print(f"\n❌ Error dropping database: {e}")
        import traceback
        traceback.print_exc()


def main():
    """Main function"""
    print("\n╔" + "="*58 + "╗")
    print("║" + " "*15 + "MongoDB Cleanup Tool" + " "*23 + "║")
    print("╚" + "="*58 + "╝")

    print("\nOptions:")
    print("1. Clear all collections (keeps structure)")
    print("2. Drop entire database (complete reset)")
    print("3. Cancel")

    choice = input("\nSelect option (1-3): ")

    if choice == '1':
        clear_database()
    elif choice == '2':
        drop_database()
    else:
        print("❌ Operation cancelled")


if __name__ == "__main__":
    main()
