import { useState, useEffect, useRef } from 'react';
import * as Haptics from 'expo-haptics';
import { useSensors } from './useSensors';
import { useLocation } from './useLocation';
import { PaveSenseML } from '../services/onnxEngine';
import { addDetection } from '../services/database';

export type EventDetection = {
  classIdx: number;
  className: string;
  confidence: number;
  potholeConfidence: number;
  timestamp: Date;
  latitude: number;
  longitude: number;
};

export const useDetection = () => {
  const { isActive, setIsActive, sensorBuffer } = useSensors();
  const { location } = useLocation();
  const [modelLoaded, setModelLoaded] = useState(false);
  const [lastEvent, setLastEvent] = useState<EventDetection | null>(null);
  const [livePotholeConf, setLivePotholeConf] = useState(0);

  const modelRef = useRef<PaveSenseML>(new PaveSenseML());
  const lastDetectionTime = useRef<number>(0);
  const DETECTION_COOLDOWN_MS = 2000;

  useEffect(() => {
    // Init ONNX Model
    modelRef.current.init().then(() => {
      setModelLoaded(true);
    });
  }, []);

  useEffect(() => {
    if (!isActive || !modelLoaded) return;

    // Run inference loop
    const interval = setInterval(async () => {
      if (sensorBuffer.isReady() && location?.coords) {
        const features = sensorBuffer.getSequence();
        if (!features) return;

        const prediction = await modelRef.current.predict(features);
        if (prediction) {
          setLivePotholeConf(prediction.potholeConfidence);

          // Pothole detection threshold
          const isPothole = prediction.className === 'pothole' && prediction.potholeConfidence > 0.6;
          
          if (isPothole) {
            const now = Date.now();
            if (now - lastDetectionTime.current > DETECTION_COOLDOWN_MS) {
              lastDetectionTime.current = now;
              
              // Trigger Haptics
              Haptics.notificationAsync(Haptics.NotificationFeedbackType.Warning);

              const eventObj = {
                classIdx: prediction.classIdx,
                className: prediction.className,
                confidence: prediction.confidence,
                potholeConfidence: prediction.potholeConfidence,
                timestamp: new Date(),
                latitude: location.coords.latitude,
                longitude: location.coords.longitude,
              };

              setLastEvent(eventObj);

              // Determine severity
              let severity = 'low';
              if (prediction.potholeConfidence > 0.85) severity = 'high';
              else if (prediction.potholeConfidence > 0.70) severity = 'medium';

              // Log to database
              await addDetection(
                eventObj.latitude,
                eventObj.longitude,
                eventObj.potholeConfidence,
                'pothole',
                severity
              );
            }
          }
        }
      }
    }, 500); // Check every 500ms for overlap

    return () => clearInterval(interval);
  }, [isActive, modelLoaded, location]);

  return {
    isActive,
    setIsActive,
    modelLoaded,
    livePotholeConf,
    lastEvent
  };
};
