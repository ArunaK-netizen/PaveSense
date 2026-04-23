"""
PaveSense FastAPI Backend
=========================
REST API for pothole detection management.
"""

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
import aiosqlite
import os
import json
from datetime import datetime, timedelta
from typing import Optional, List
from pydantic import BaseModel, Field
import math

# === Configuration ===
DB_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "database")
DB_FILE = os.path.join(DB_DIR, "pavesense.db")
LOCATION_DISTANCE_THRESHOLD = 0.00005  # ~5 meters


# === Pydantic Schemas ===

class DetectionCreate(BaseModel):
    latitude: float = Field(..., ge=-90, le=90)
    longitude: float = Field(..., ge=-180, le=180)
    confidence: float = Field(..., ge=0, le=1)
    severity: str = Field(default="medium", pattern="^(low|medium|high)$")
    event_class: str = Field(default="pothole")
    speed_kmh: Optional[float] = None
    device_id: Optional[str] = None
    accel_magnitude: Optional[float] = None
    gyro_magnitude: Optional[float] = None

class DetectionResponse(BaseModel):
    id: int
    latitude: float
    longitude: float
    confidence: float
    severity: str
    event_class: str
    speed_kmh: Optional[float]
    device_id: Optional[str]
    confirmations: int
    rejections: int
    timestamp: str

class DetectionBatch(BaseModel):
    detections: List[DetectionCreate]

class UserCreate(BaseModel):
    device_id: str
    display_name: Optional[str] = None

class UserStats(BaseModel):
    device_id: str
    display_name: Optional[str]
    total_detections: int
    confirmed_detections: int
    total_distance_km: float
    joined_date: str

class StatsResponse(BaseModel):
    total_detections: int
    high_severity: int
    medium_severity: int
    low_severity: int
    confirmed: int
    rejected: int
    unique_devices: int
    last_24h: int
    last_7d: int

class ManualReport(BaseModel):
    latitude: float = Field(..., ge=-90, le=90)
    longitude: float = Field(..., ge=-180, le=180)
    severity: str = Field(default="medium", pattern="^(low|medium|high)$")
    description: Optional[str] = None
    device_id: Optional[str] = None


# === Database Setup ===

async def init_db():
    os.makedirs(DB_DIR, exist_ok=True)
    async with aiosqlite.connect(DB_FILE) as db:
        await db.execute("""
            CREATE TABLE IF NOT EXISTS detections (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                latitude REAL NOT NULL,
                longitude REAL NOT NULL,
                confidence REAL NOT NULL DEFAULT 0.0,
                severity TEXT NOT NULL DEFAULT 'medium',
                event_class TEXT NOT NULL DEFAULT 'pothole',
                speed_kmh REAL,
                device_id TEXT,
                accel_magnitude REAL,
                gyro_magnitude REAL,
                confirmations INTEGER DEFAULT 0,
                rejections INTEGER DEFAULT 0,
                is_manual INTEGER DEFAULT 0,
                description TEXT,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
            )
        """)
        await db.execute("""
            CREATE TABLE IF NOT EXISTS users (
                device_id TEXT PRIMARY KEY,
                display_name TEXT,
                total_detections INTEGER DEFAULT 0,
                confirmed_detections INTEGER DEFAULT 0,
                total_distance_km REAL DEFAULT 0.0,
                joined_date DATETIME DEFAULT CURRENT_TIMESTAMP
            )
        """)
        await db.execute("""
            CREATE INDEX IF NOT EXISTS idx_detections_location
            ON detections(latitude, longitude)
        """)
        await db.execute("""
            CREATE INDEX IF NOT EXISTS idx_detections_timestamp
            ON detections(timestamp)
        """)
        await db.commit()


# === FastAPI App ===

@asynccontextmanager
async def lifespan(app: FastAPI):
    await init_db()
    print("[PaveSense Backend] Database initialized")
    yield

app = FastAPI(
    title="PaveSense API",
    description="AI-powered road intelligence platform API",
    version="2.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# === Helper Functions ===

async def is_duplicate(db, lat, lng, threshold=LOCATION_DISTANCE_THRESHOLD):
    cursor = await db.execute(
        "SELECT 1 FROM detections WHERE ABS(latitude - ?) < ? AND ABS(longitude - ?) < ? LIMIT 1",
        (lat, threshold, lng, threshold),
    )
    return await cursor.fetchone() is not None


def confidence_to_severity(confidence: float) -> str:
    if confidence >= 0.8:
        return "high"
    elif confidence >= 0.5:
        return "medium"
    return "low"


def haversine_distance(lat1, lon1, lat2, lon2):
    """Distance in km between two GPS points."""
    R = 6371
    dlat = math.radians(lat2 - lat1)
    dlon = math.radians(lon2 - lon1)
    a = math.sin(dlat/2)**2 + math.cos(math.radians(lat1)) * math.cos(math.radians(lat2)) * math.sin(dlon/2)**2
    return R * 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))


# === Detection Endpoints ===

@app.post("/api/detections", response_model=DetectionResponse)
async def create_detection(detection: DetectionCreate):
    """Store a new pothole detection event."""
    severity = detection.severity or confidence_to_severity(detection.confidence)

    async with aiosqlite.connect(DB_FILE) as db:
        # Check for duplicate
        if await is_duplicate(db, detection.latitude, detection.longitude):
            raise HTTPException(status_code=409, detail="Duplicate detection within 5m radius")

        cursor = await db.execute(
            """INSERT INTO detections 
               (latitude, longitude, confidence, severity, event_class, speed_kmh, 
                device_id, accel_magnitude, gyro_magnitude)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (detection.latitude, detection.longitude, detection.confidence,
             severity, detection.event_class, detection.speed_kmh,
             detection.device_id, detection.accel_magnitude, detection.gyro_magnitude),
        )
        await db.commit()
        detection_id = cursor.lastrowid

        # Update user stats
        if detection.device_id:
            await db.execute(
                """INSERT INTO users (device_id, total_detections) VALUES (?, 1)
                   ON CONFLICT(device_id) DO UPDATE SET total_detections = total_detections + 1""",
                (detection.device_id,),
            )
            await db.commit()

        return DetectionResponse(
            id=detection_id,
            latitude=detection.latitude,
            longitude=detection.longitude,
            confidence=detection.confidence,
            severity=severity,
            event_class=detection.event_class,
            speed_kmh=detection.speed_kmh,
            device_id=detection.device_id,
            confirmations=0,
            rejections=0,
            timestamp=datetime.utcnow().isoformat(),
        )


@app.post("/api/detections/batch")
async def create_detections_batch(batch: DetectionBatch):
    """Store multiple detection events (for offline sync)."""
    created = 0
    skipped = 0

    async with aiosqlite.connect(DB_FILE) as db:
        for det in batch.detections:
            if await is_duplicate(db, det.latitude, det.longitude):
                skipped += 1
                continue

            severity = det.severity or confidence_to_severity(det.confidence)
            await db.execute(
                """INSERT INTO detections 
                   (latitude, longitude, confidence, severity, event_class, speed_kmh, 
                    device_id, accel_magnitude, gyro_magnitude)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (det.latitude, det.longitude, det.confidence, severity,
                 det.event_class, det.speed_kmh, det.device_id,
                 det.accel_magnitude, det.gyro_magnitude),
            )
            created += 1

        await db.commit()

    return {"created": created, "skipped": skipped, "total": len(batch.detections)}


@app.get("/api/detections")
async def get_detections(
    severity: Optional[str] = Query(None, pattern="^(low|medium|high)$"),
    hours: Optional[int] = Query(None, ge=1),
    confirmed_only: bool = Query(False),
    lat_min: Optional[float] = None,
    lat_max: Optional[float] = None,
    lng_min: Optional[float] = None,
    lng_max: Optional[float] = None,
    limit: int = Query(500, ge=1, le=5000),
):
    """Fetch detections with optional filters."""
    query = "SELECT * FROM detections WHERE 1=1"
    params = []

    if severity:
        query += " AND severity = ?"
        params.append(severity)

    if hours:
        cutoff = (datetime.utcnow() - timedelta(hours=hours)).isoformat()
        query += " AND timestamp >= ?"
        params.append(cutoff)

    if confirmed_only:
        query += " AND confirmations > rejections"

    if lat_min is not None and lat_max is not None:
        query += " AND latitude BETWEEN ? AND ?"
        params.extend([lat_min, lat_max])

    if lng_min is not None and lng_max is not None:
        query += " AND longitude BETWEEN ? AND ?"
        params.extend([lng_min, lng_max])

    query += " ORDER BY timestamp DESC LIMIT ?"
    params.append(limit)

    async with aiosqlite.connect(DB_FILE) as db:
        db.row_factory = aiosqlite.Row
        cursor = await db.execute(query, params)
        rows = await cursor.fetchall()

        return [
            {
                "id": row["id"],
                "latitude": row["latitude"],
                "longitude": row["longitude"],
                "confidence": row["confidence"],
                "severity": row["severity"],
                "event_class": row["event_class"],
                "speed_kmh": row["speed_kmh"],
                "device_id": row["device_id"],
                "confirmations": row["confirmations"],
                "rejections": row["rejections"],
                "timestamp": row["timestamp"],
            }
            for row in rows
        ]


@app.get("/api/detections/nearby")
async def get_nearby_detections(
    lat: float = Query(..., ge=-90, le=90),
    lng: float = Query(..., ge=-180, le=180),
    radius_km: float = Query(1.0, ge=0.1, le=50),
    limit: int = Query(100, ge=1, le=1000),
):
    """Get detections within a radius of a point."""
    # Approximate degree offset for the radius
    lat_offset = radius_km / 111.0
    lng_offset = radius_km / (111.0 * math.cos(math.radians(lat)))

    async with aiosqlite.connect(DB_FILE) as db:
        db.row_factory = aiosqlite.Row
        cursor = await db.execute(
            """SELECT * FROM detections 
               WHERE latitude BETWEEN ? AND ? 
               AND longitude BETWEEN ? AND ?
               ORDER BY timestamp DESC LIMIT ?""",
            (lat - lat_offset, lat + lat_offset,
             lng - lng_offset, lng + lng_offset, limit),
        )
        rows = await cursor.fetchall()

        results = []
        for row in rows:
            dist = haversine_distance(lat, lng, row["latitude"], row["longitude"])
            if dist <= radius_km:
                results.append({
                    "id": row["id"],
                    "latitude": row["latitude"],
                    "longitude": row["longitude"],
                    "confidence": row["confidence"],
                    "severity": row["severity"],
                    "distance_km": round(dist, 3),
                    "confirmations": row["confirmations"],
                    "rejections": row["rejections"],
                    "timestamp": row["timestamp"],
                })

        return sorted(results, key=lambda x: x["distance_km"])


@app.put("/api/detections/{detection_id}/confirm")
async def confirm_detection(detection_id: int, device_id: Optional[str] = None):
    """User confirms a pothole exists."""
    async with aiosqlite.connect(DB_FILE) as db:
        cursor = await db.execute(
            "UPDATE detections SET confirmations = confirmations + 1 WHERE id = ?",
            (detection_id,),
        )
        if cursor.rowcount == 0:
            raise HTTPException(status_code=404, detail="Detection not found")
        await db.commit()

        if device_id:
            await db.execute(
                """INSERT INTO users (device_id, confirmed_detections) VALUES (?, 1)
                   ON CONFLICT(device_id) DO UPDATE SET confirmed_detections = confirmed_detections + 1""",
                (device_id,),
            )
            await db.commit()

    return {"status": "confirmed", "detection_id": detection_id}


@app.put("/api/detections/{detection_id}/reject")
async def reject_detection(detection_id: int):
    """User rejects a detection (not a real pothole)."""
    async with aiosqlite.connect(DB_FILE) as db:
        cursor = await db.execute(
            "UPDATE detections SET rejections = rejections + 1 WHERE id = ?",
            (detection_id,),
        )
        if cursor.rowcount == 0:
            raise HTTPException(status_code=404, detail="Detection not found")
        await db.commit()

    return {"status": "rejected", "detection_id": detection_id}


@app.post("/api/detections/report")
async def manual_report(report: ManualReport):
    """Manually report a pothole location."""
    async with aiosqlite.connect(DB_FILE) as db:
        cursor = await db.execute(
            """INSERT INTO detections 
               (latitude, longitude, confidence, severity, event_class, 
                device_id, is_manual, description)
               VALUES (?, ?, 1.0, ?, 'pothole', ?, 1, ?)""",
            (report.latitude, report.longitude, report.severity,
             report.device_id, report.description),
        )
        await db.commit()

    return {"status": "reported", "id": cursor.lastrowid}


# === Stats Endpoint ===

@app.get("/api/stats", response_model=StatsResponse)
async def get_stats():
    """Get analytics summary."""
    async with aiosqlite.connect(DB_FILE) as db:
        total = (await (await db.execute("SELECT COUNT(*) FROM detections")).fetchone())[0]
        high = (await (await db.execute("SELECT COUNT(*) FROM detections WHERE severity='high'")).fetchone())[0]
        medium = (await (await db.execute("SELECT COUNT(*) FROM detections WHERE severity='medium'")).fetchone())[0]
        low = (await (await db.execute("SELECT COUNT(*) FROM detections WHERE severity='low'")).fetchone())[0]
        confirmed = (await (await db.execute("SELECT COUNT(*) FROM detections WHERE confirmations > 0")).fetchone())[0]
        rejected = (await (await db.execute("SELECT COUNT(*) FROM detections WHERE rejections > confirmations")).fetchone())[0]
        devices = (await (await db.execute("SELECT COUNT(DISTINCT device_id) FROM detections WHERE device_id IS NOT NULL")).fetchone())[0]

        cutoff_24h = (datetime.utcnow() - timedelta(hours=24)).isoformat()
        last_24h = (await (await db.execute("SELECT COUNT(*) FROM detections WHERE timestamp >= ?", (cutoff_24h,))).fetchone())[0]

        cutoff_7d = (datetime.utcnow() - timedelta(days=7)).isoformat()
        last_7d = (await (await db.execute("SELECT COUNT(*) FROM detections WHERE timestamp >= ?", (cutoff_7d,))).fetchone())[0]

        return StatsResponse(
            total_detections=total,
            high_severity=high,
            medium_severity=medium,
            low_severity=low,
            confirmed=confirmed,
            rejected=rejected,
            unique_devices=devices,
            last_24h=last_24h,
            last_7d=last_7d,
        )


# === User Endpoints ===

@app.post("/api/users")
async def create_or_update_user(user: UserCreate):
    """Create or update user profile."""
    async with aiosqlite.connect(DB_FILE) as db:
        await db.execute(
            """INSERT INTO users (device_id, display_name) VALUES (?, ?)
               ON CONFLICT(device_id) DO UPDATE SET display_name = ?""",
            (user.device_id, user.display_name, user.display_name),
        )
        await db.commit()

    return {"status": "ok", "device_id": user.device_id}


@app.get("/api/users/{device_id}/stats")
async def get_user_stats(device_id: str):
    """Get user contribution stats."""
    async with aiosqlite.connect(DB_FILE) as db:
        db.row_factory = aiosqlite.Row
        cursor = await db.execute("SELECT * FROM users WHERE device_id = ?", (device_id,))
        row = await cursor.fetchone()

        if not row:
            raise HTTPException(status_code=404, detail="User not found")

        return UserStats(
            device_id=row["device_id"],
            display_name=row["display_name"],
            total_detections=row["total_detections"],
            confirmed_detections=row["confirmed_detections"],
            total_distance_km=row["total_distance_km"],
            joined_date=row["joined_date"],
        )


# === Health Check ===

@app.get("/api/health")
async def health_check():
    return {"status": "ok", "service": "PaveSense API", "version": "2.0.0"}
