from sqlalchemy import text
from app.database import engine

def run_database_migrations():
    """
    Safely adds any missing columns to existing SQLite database tables
    without dropping or modifying existing records.
    Also migrates 'image_analyses.section_id' to nullable so that
    analyses can be stored without a mandatory section association,
    and adds 'user_id' and 'analysis_mode' to 'image_analyses'.
    """
    with engine.connect() as conn:
        # 1. Check columns in 'estates'
        cursor = conn.execute(text("PRAGMA table_info(estates)"))
        existing_estate_cols = [row[1] for row in cursor.fetchall()]

        estate_cols_to_add = [
            ("total_hectares", "FLOAT"),
            ("area_unit", "VARCHAR(20) DEFAULT 'acres'"),
            ("description", "TEXT"),
            ("boundary_geojson", "TEXT"),
        ]
        for col_name, col_type in estate_cols_to_add:
            if col_name not in existing_estate_cols:
                try:
                    conn.execute(text(f"ALTER TABLE estates ADD COLUMN {col_name} {col_type}"))
                    print(f"[Migration] Added column '{col_name}' to 'estates'")
                except Exception as e:
                    print(f"[Migration] Notice on estates.{col_name}: {e}")

        # 2. Check columns in 'sections'
        cursor = conn.execute(text("PRAGMA table_info(sections)"))
        existing_section_cols = [row[1] for row in cursor.fetchall()]

        section_cols_to_add = [
            ("area_hectares", "FLOAT"),
            ("color", "VARCHAR(20) DEFAULT '#15803d'"),
            ("boundary_geojson", "TEXT"),
        ]
        for col_name, col_type in section_cols_to_add:
            if col_name not in existing_section_cols:
                try:
                    conn.execute(text(f"ALTER TABLE sections ADD COLUMN {col_name} {col_type}"))
                    print(f"[Migration] Added column '{col_name}' to 'sections'")
                except Exception as e:
                    print(f"[Migration] Notice on sections.{col_name}: {e}")

        # 3. Migrate 'image_analyses.section_id' to nullable if it is currently NOT NULL.
        cursor = conn.execute(text("PRAGMA table_info(image_analyses)"))
        analysis_info = {row[1]: row for row in cursor.fetchall()}
        if "section_id" in analysis_info and analysis_info["section_id"][3] == 1:
            try:
                conn.execute(text("PRAGMA foreign_keys=OFF"))
                conn.execute(text("""
                    CREATE TABLE image_analyses_new (
                        id INTEGER NOT NULL PRIMARY KEY,
                        user_id INTEGER,
                        section_id INTEGER,
                        original_filename VARCHAR(255) NOT NULL,
                        storage_path VARCHAR(500) NOT NULL,
                        annotated_storage_path VARCHAR(500),
                        uploaded_at DATETIME,
                        analyzed_at DATETIME,
                        analysis_mode VARCHAR(50) DEFAULT 'branch',
                        model_version VARCHAR(100),
                        model_status_at_inference VARCHAR(100),
                        inference_time_ms FLOAT,
                        confidence_threshold FLOAT,
                        total_cherries INTEGER,
                        dark_brown_count INTEGER,
                        green_count INTEGER,
                        low_visibility_count INTEGER,
                        red_count INTEGER,
                        yellow_count INTEGER,
                        dark_brown_pct FLOAT,
                        green_pct FLOAT,
                        low_visibility_pct FLOAT,
                        red_pct FLOAT,
                        yellow_pct FLOAT,
                        maturity_stage VARCHAR(50),
                        maturity_notes TEXT,
                        detections_json TEXT,
                        is_sample_data BOOLEAN,
                        FOREIGN KEY (section_id) REFERENCES sections (id) ON DELETE CASCADE,
                        FOREIGN KEY (user_id) REFERENCES users (id) ON DELETE CASCADE
                    )
                """))
                conn.execute(text("INSERT INTO image_analyses_new SELECT id, NULL, section_id, original_filename, storage_path, annotated_storage_path, uploaded_at, analyzed_at, 'branch', model_version, model_status_at_inference, inference_time_ms, confidence_threshold, total_cherries, dark_brown_count, green_count, low_visibility_count, red_count, yellow_count, dark_brown_pct, green_pct, low_visibility_pct, red_pct, yellow_pct, maturity_stage, maturity_notes, detections_json, is_sample_data FROM image_analyses"))
                conn.execute(text("DROP TABLE image_analyses"))
                conn.execute(text("ALTER TABLE image_analyses_new RENAME TO image_analyses"))
                conn.execute(text("CREATE INDEX IF NOT EXISTS ix_image_analyses_id ON image_analyses (id)"))
                conn.execute(text("CREATE INDEX IF NOT EXISTS ix_image_analyses_user_id ON image_analyses (user_id)"))
                conn.execute(text("PRAGMA foreign_keys=ON"))
                print("[Migration] Migrated 'image_analyses.section_id' to nullable and added 'user_id'")
            except Exception as e:
                print(f"[Migration] Notice on image_analyses rebuild: {e}")
        else:
            # Check if user_id column exists in image_analyses
            if "user_id" not in analysis_info:
                try:
                    conn.execute(text("ALTER TABLE image_analyses ADD COLUMN user_id INTEGER REFERENCES users (id) ON DELETE CASCADE"))
                    conn.execute(text("CREATE INDEX IF NOT EXISTS ix_image_analyses_user_id ON image_analyses (user_id)"))
                    print("[Migration] Added 'user_id' column to 'image_analyses'")
                except Exception as e:
                    print(f"[Migration] Notice on adding user_id: {e}")

        # 4. Check for 'analysis_mode' in 'image_analyses'
        cursor = conn.execute(text("PRAGMA table_info(image_analyses)"))
        current_analysis_cols = [row[1] for row in cursor.fetchall()]
        if "analysis_mode" not in current_analysis_cols:
            try:
                conn.execute(text("ALTER TABLE image_analyses ADD COLUMN analysis_mode VARCHAR(50) DEFAULT 'branch'"))
                print("[Migration] Added column 'analysis_mode' to 'image_analyses'")
            except Exception as e:
                print(f"[Migration] Notice on image_analyses.analysis_mode: {e}")

        # 4. Check for 'analysis_mode' in 'image_analyses'
        cursor = conn.execute(text("PRAGMA table_info(image_analyses)"))
        current_analysis_cols = [row[1] for row in cursor.fetchall()]
        if "analysis_mode" not in current_analysis_cols:
            try:
                conn.execute(text("ALTER TABLE image_analyses ADD COLUMN analysis_mode VARCHAR(50) DEFAULT 'branch'"))
                print("[Migration] Added column 'analysis_mode' to 'image_analyses'")
            except Exception as e:
                print(f"[Migration] Notice on image_analyses.analysis_mode: {e}")

        conn.commit()
