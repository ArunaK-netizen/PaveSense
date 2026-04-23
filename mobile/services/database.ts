import * as SQLite from 'expo-sqlite';

// Open the database securely
let dbSync: SQLite.SQLiteDatabase;

export const initDB = async () => {
  dbSync = await SQLite.openDatabaseAsync('pavesense.db');
  
  await dbSync.execAsync(`
    PRAGMA journal_mode = WAL;
    CREATE TABLE IF NOT EXISTS detections (
      id INTEGER PRIMARY KEY AUTOINCREMENT,
      latitude REAL NOT NULL,
      longitude REAL NOT NULL,
      confidence REAL NOT NULL,
      severity TEXT NOT NULL,
      category TEXT NOT NULL,
      timestamp DATETIME DEFAULT (datetime('now', 'localtime'))
    );
  `);
};

export const addDetection = async (
  lat: number,
  lng: number,
  confidence: number,
  category: string,
  severity: string
) => {
  if (!dbSync) return null;
  try {
    // Basic deduplication check: within ~50 meters in the last hour
    const latThreshold = 0.0005; // approx 50m
    const lngThreshold = 0.0005;

    const recentDetections = await dbSync.getAllAsync(`
      SELECT id FROM detections 
      WHERE ABS(latitude - ?) < ? 
        AND ABS(longitude - ?) < ? 
        AND timestamp > datetime('now', '-1 hour')
    `, [lat, latThreshold, lng, lngThreshold]);

    if (recentDetections && recentDetections.length > 0) {
      console.log('Skipping duplicate detection');
      return false; // deduplicated locally
    }

    const result = await dbSync.runAsync(`
      INSERT INTO detections (latitude, longitude, confidence, severity, category)
      VALUES (?, ?, ?, ?, ?)
    `, [lat, lng, confidence, severity, category]);

    return result.lastInsertRowId;
  } catch (err) {
    console.error('DB Insert error:', err);
    return null;
  }
};

export const getDetections = async () => {
  if (!dbSync) return [];
  try {
    const records = await dbSync.getAllAsync('SELECT * FROM detections ORDER BY timestamp DESC LIMIT 100');
    return records as any[];
  } catch (err) {
    console.error('DB Fetch error:', err);
    return [];
  }
};

export const clearDetections = async () => {
  if (!dbSync) return;
  await dbSync.runAsync('DELETE FROM detections;');
};
