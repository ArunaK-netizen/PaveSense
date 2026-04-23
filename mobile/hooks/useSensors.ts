import { useState, useEffect, useRef } from 'react';
import { Accelerometer, Gyroscope } from 'expo-sensors';
import { SensorBuffer } from '../services/sensorBuffer';

const SAMPLING_INTERVAL_MS = 20; // 50 Hz

export const useSensors = () => {
  const [isActive, setIsActive] = useState(false);
  const bufferRef = useRef(new SensorBuffer());
  
  // We explicitly keep track of the latest values to sample ourselves
  // since expo-sensors callback frequencies can be inconsistent.
  const latestAccel = useRef({ x: 0, y: 0, z: 0 });
  const latestGyro = useRef({ x: 0, y: 0, z: 0 });
  const samplingTimer = useRef<NodeJS.Timeout | null>(null);

  useEffect(() => {
    Accelerometer.setUpdateInterval(15); // slightly faster to ensure freshness
    Gyroscope.setUpdateInterval(15);
    
    let accelSub: ReturnType<typeof Accelerometer.addListener> | null = null;
    let gyroSub: ReturnType<typeof Gyroscope.addListener> | null = null;

    if (isActive) {
      accelSub = Accelerometer.addListener(data => {
        latestAccel.current = data;
      });
      gyroSub = Gyroscope.addListener(data => {
        latestGyro.current = data;
      });

      samplingTimer.current = setInterval(() => {
        // Collect sample at regular 50Hz interval
        bufferRef.current.addSample(
          latestAccel.current,
          latestGyro.current,
          Date.now()
        );
      }, SAMPLING_INTERVAL_MS);

    } else {
      if (accelSub) accelSub.remove();
      if (gyroSub) gyroSub.remove();
      if (samplingTimer.current) clearInterval(samplingTimer.current);
      bufferRef.current.clear();
    }

    return () => {
      if (accelSub) accelSub.remove();
      if (gyroSub) gyroSub.remove();
      if (samplingTimer.current) clearInterval(samplingTimer.current);
    };
  }, [isActive]);

  return {
    isActive,
    setIsActive,
    sensorBuffer: bufferRef.current,
  };
};
