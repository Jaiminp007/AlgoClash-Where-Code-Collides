"""
MongoDB Connection Manager
"""

from pymongo import MongoClient, ASCENDING, DESCENDING
from pymongo.errors import ConnectionFailure, ServerSelectionTimeoutError
import os
import logging

logger = logging.getLogger(__name__)

class MongoDB:
    """MongoDB connection manager singleton"""

    _instance = None
    _client = None
    _db = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(MongoDB, cls).__new__(cls)
        return cls._instance

    def __init__(self):
        if not hasattr(self, 'initialized'):
            self.initialized = False

    def connect(self, uri=None, db_name=None):
        """Connect to MongoDB"""
        if self.initialized:
            return self._db

        uri = uri or os.getenv('MONGODB_URI', 'mongodb://localhost:27017/ai_trader_battlefield')
        db_name = db_name or os.getenv('MONGODB_DATABASE', 'ai_trader_battlefield')

        try:
            self._client = MongoClient(
                uri,
                serverSelectionTimeoutMS=5000,
                connectTimeoutMS=10000,
                maxPoolSize=50
            )

            # Test connection
            self._client.admin.command('ping')
            self._db = self._client[db_name]

            # Create indexes
            self._create_indexes()

            self.initialized = True
            print(f"✅ Connected to MongoDB: {db_name}")
            logger.info(f"✅ Connected to MongoDB: {db_name}")
            return self._db

        except Exception as e:
            print(f"❌ Failed to connect to MongoDB: {e}")
            logger.error(f"❌ Failed to connect to MongoDB: {e}")
            raise

    def _create_indexes(self):
        """Create database indexes"""
        try:
            # Simulations
            self._db.simulations.create_index('simulation_id', unique=True)
            self._db.simulations.create_index([('created_at', DESCENDING)])
            self._db.simulations.create_index('status')

            # Generations
            self._db.generations.create_index('generation_id', unique=True)
            self._db.generations.create_index([('created_at', DESCENDING)])

            # Algorithms
            self._db.algorithms.create_index([('simulation_id', ASCENDING), ('model_name', ASCENDING)])
            self._db.algorithms.create_index('code_hash')

            # Ticks
            self._db.simulation_ticks.create_index([('simulation_id', ASCENDING), ('tick_number', ASCENDING)])

            print("✅ Database indexes created")
            logger.info("✅ Database indexes created")
        except Exception as e:
            logger.warning(f"⚠️ Error creating indexes: {e}")

    def get_db(self):
        """Get database instance"""
        if not self.initialized:
            return self.connect()
        return self._db

    def close(self):
        """Close connection"""
        if self._client:
            self._client.close()
            self.initialized = False
            self._db = None
            print("MongoDB connection closed")

# Global instance
_mongodb = MongoDB()

def init_db(uri=None, db_name=None):
    """Initialize MongoDB"""
    return _mongodb.connect(uri, db_name)

def get_db():
    """Get database"""
    return _mongodb.get_db()

def close_db():
    """Close database"""
    _mongodb.close()
