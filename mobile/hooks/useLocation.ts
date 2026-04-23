import { useState, useEffect } from 'react';
import * as Location from 'expo-location';

export const useLocation = () => {
  const [location, setLocation] = useState<Location.LocationObject | null>(null);
  const [errorMsg, setErrorMsg] = useState<string | null>(null);

  useEffect(() => {
    let subscriber: Location.LocationSubscription | null = null;
    
    (async () => {
      let { status } = await Location.requestForegroundPermissionsAsync();
      if (status !== 'granted') {
        setErrorMsg('Permission to access location was denied');
        return;
      }

      subscriber = await Location.watchPositionAsync(
        {
          accuracy: Location.Accuracy.High,
          distanceInterval: 1, // update every 1 meter
          timeInterval: 1000,   // or every 1 second
        },
        (loc) => {
          setLocation(loc);
        }
      );
    })();

    return () => {
      if (subscriber) {
        subscriber.remove();
      }
    };
  }, []);

  return { location, errorMsg };
};
