import sqlite3
import time
from datetime import datetime
from pathlib import Path
import difflib

class TollDatabase:
    """SQLite Database manager for RDK X5 ANPR Toll Booth System."""
    
    def __init__(self, db_path: str | Path = None):
        if db_path is None:
            base_dir = Path(__file__).resolve().parent.parent
            self.db_path = base_dir / "data" / "anpr.db"
        else:
            self.db_path = Path(db_path)
            
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.seen_plates = {}  # In-memory deduplication cache: {plate: last_seen_timestamp}
        self.init_db()
        
    def get_connection(self) -> sqlite3.Connection:
        conn = sqlite3.connect(str(self.db_path), check_same_thread=False)
        conn.row_factory = sqlite3.Row
        return conn

    def init_db(self):
        """Create database tables if they do not exist."""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS vehicle_records (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    plate_number TEXT NOT NULL,
                    timestamp TEXT NOT NULL,
                    date_str TEXT NOT NULL,
                    time_str TEXT NOT NULL,
                    yolo_conf REAL NOT NULL,
                    ocr_conf REAL NOT NULL,
                    toll_amount INTEGER NOT NULL DEFAULT 100,
                    crop_filename TEXT,
                    source TEXT,
                    raw_text TEXT
                )
            """)
            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_timestamp ON vehicle_records(timestamp);
            """)
            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_plate ON vehicle_records(plate_number);
            """)
            conn.commit()

    def is_duplicate(self, plate_number: str, current_time_sec: float, cooldown_sec: float = 10.0, similarity_thresh: float = 0.7) -> bool:
        """Check if plate is a duplicate within cooldown window using string similarity."""
        for seen_plate, last_seen_time in list(self.seen_plates.items()):
            if (current_time_sec - last_seen_time) <= cooldown_sec:
                sim = difflib.SequenceMatcher(None, plate_number, seen_plate).ratio()
                if sim >= similarity_thresh:
                    # Refresh cooldown for original plate
                    self.seen_plates[seen_plate] = current_time_sec
                    return True
                    
        return False

    def insert_record(self, plate_number: str, yolo_conf: float, ocr_conf: float, 
                      raw_text: str = "", crop_filename: str = "", source: str = "gs130w", 
                      toll_amount: int = 100, current_time_sec: float = None) -> int:
        """Insert a new vehicle toll record into SQLite."""
        now = datetime.now()
        ts_str = now.strftime("%Y-%m-%d %H:%M:%S")
        date_str = now.strftime("%Y-%m-%d")
        time_str = now.strftime("%H:%M:%S")
        
        # Update in-memory cache
        sec = current_time_sec if current_time_sec is not None else time.time()
        self.seen_plates[plate_number] = sec

        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                INSERT INTO vehicle_records (
                    plate_number, timestamp, date_str, time_str, 
                    yolo_conf, ocr_conf, toll_amount, crop_filename, source, raw_text
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (plate_number, ts_str, date_str, time_str, yolo_conf, ocr_conf, toll_amount, crop_filename, source, raw_text))
            conn.commit()
            return cursor.lastrowid

    def get_recent(self, limit: int = 20) -> list[dict]:
        """Fetch the most recent vehicle records for the web dashboard."""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT id, plate_number, timestamp, date_str, time_str, 
                       yolo_conf, ocr_conf, toll_amount, crop_filename, source, raw_text
                FROM vehicle_records 
                ORDER BY id DESC 
                LIMIT ?
            """, (limit,))
            rows = cursor.fetchall()
            return [dict(row) for row in rows]

    def get_stats(self) -> dict:
        """Return summary statistics for the toll booth dashboard."""
        today_str = datetime.now().strftime("%Y-%m-%d")
        with self.get_connection() as conn:
            cursor = conn.cursor()
            
            # All-time stats
            cursor.execute("SELECT COUNT(*), COALESCE(SUM(toll_amount), 0) FROM vehicle_records")
            total_vehicles, total_toll = cursor.fetchone()
            
            # Today's stats
            cursor.execute("SELECT COUNT(*), COALESCE(SUM(toll_amount), 0) FROM vehicle_records WHERE date_str = ?", (today_str,))
            today_vehicles, today_toll = cursor.fetchone()
            
            return {
                "total_vehicles": total_vehicles,
                "total_toll": total_toll,
                "today_vehicles": today_vehicles,
                "today_toll": today_toll,
                "system_status": "Online",
                "last_updated": datetime.now().strftime("%H:%M:%S")
            }
