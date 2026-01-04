import logging
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))

from utils.db_utils import (
    create_database_if_not_exists,
    initialize_schema,
    load_csv_to_table
)

logging.basicConfig(
    level=logging.INFO,
    format='%(levelname)s:%(name)s:%(message)s'
)
logger = logging.getLogger(__name__)


def initialize_water_quality_db():
    try:
        logger.info("Initializing water quality database")

        create_database_if_not_exists("water_quality_db")

        project_root = Path(__file__).parent.parent.parent
        schema_path = project_root / 'sql' / 'init_schema.sql'

        if not schema_path.exists():
            raise FileNotFoundError(f"Schema file not found: {schema_path}")

        initialize_schema(str(schema_path), "water_quality_db")

        csv_path = project_root / 'data' / 'sample' / 'water_potability.csv'

        if csv_path.exists():
            rows = load_csv_to_table(
                str(csv_path),
                "raw_water_quality",
                db_name="water_quality_db",
                if_exists="replace"
            )
            logger.info(f"Loaded {rows} rows to raw_water_quality")
        else:
            logger.warning(f"CSV not found: {csv_path}")

        logger.info("Database initialization complete")
        return True

    except Exception as e:
        logger.error(f"Initialization failed: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = initialize_water_quality_db()
    sys.exit(0 if success else 1)
