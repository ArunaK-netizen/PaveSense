import { extractFeatures } from './featureEngine';

const SEQUENCE_LENGTH = 100;

export class SensorBuffer {
  private accel = {
    x: [] as number[],
    y: [] as number[],
    z: [] as number[],
  };
  
  private gyro = {
    x: [] as number[],
    y: [] as number[],
    z: [] as number[],
  };

  private lastAddTimestamp = 0;

  addSample(a: {x: number, y: number, z: number}, g: {x: number, y: number, z: number}, timestamp: number) {
    // Only accept at rough 50Hz (20ms interval) to avoid jitter buildup,
    // though expo-sensors timestamp is slightly variable
    const dt = timestamp - this.lastAddTimestamp;
    if (this.lastAddTimestamp !== 0 && dt < 10) return false; 
    
    this.lastAddTimestamp = timestamp;

    this.accel.x.push(a.x * 9.81); // Convert Gs to m/s^2 (expo-sensors returns Gs)
    this.accel.y.push(a.y * 9.81);
    this.accel.z.push(a.z * 9.81);

    this.gyro.x.push(g.x);
    this.gyro.y.push(g.y);
    this.gyro.z.push(g.z);

    // Keep ring buffer size bounded
    if (this.accel.x.length > SEQUENCE_LENGTH) {
      this.accel.x.shift();
      this.accel.y.shift();
      this.accel.z.shift();
      
      this.gyro.x.shift();
      this.gyro.y.shift();
      this.gyro.z.shift();
    }
    return true;
  }

  isReady() {
    return this.accel.x.length === SEQUENCE_LENGTH;
  }

  getSequence() {
    if (!this.isReady()) return null;
    
    // Pass copies
    return extractFeatures(
      {x: [...this.accel.x], y: [...this.accel.y], z: [...this.accel.z]},
      {x: [...this.gyro.x], y: [...this.gyro.y], z: [...this.gyro.z]}
    );
  }

  clear() {
    this.accel = { x: [], y: [], z: [] };
    this.gyro = { x: [], y: [], z: [] };
    this.lastAddTimestamp = 0;
  }
}
