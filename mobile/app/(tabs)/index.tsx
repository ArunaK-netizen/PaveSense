import React, { useEffect, useState } from 'react';
import { StyleSheet, View, Text, Switch, Dimensions } from 'react-native';
import MapView, { Marker, Polyline } from 'react-native-maps';
import { useDetection } from '../../hooks/useDetection';
import { useLocation } from '../../hooks/useLocation';
import { Theme } from '../../constants/theme';
import { getDetections } from '../../services/database';

const { width } = Dimensions.get('window');

export default function TabOneScreen() {
  const { isActive, setIsActive, modelLoaded, livePotholeConf, lastEvent } = useDetection();
  const { location } = useLocation();
  const [historicDetections, setHistoricDetections] = useState<any[]>([]);

  // Periodically fetch from DB
  useEffect(() => {
    const fetchDetections = async () => {
      const docs = await getDetections();
      setHistoricDetections(docs);
    };
    fetchDetections();
    const interval = setInterval(fetchDetections, 5000);
    return () => clearInterval(interval);
  }, []);

  const getMarkerColor = (severity: string) => {
    switch (severity) {
      case 'high': return Theme.colors.markerHigh;
      case 'medium': return Theme.colors.markerMedium;
      default: return Theme.colors.markerLow;
    }
  };

  return (
    <View style={styles.container}>
      {location?.coords ? (
        <MapView
          style={styles.map}
          initialRegion={{
            latitude: location.coords.latitude,
            longitude: location.coords.longitude,
            latitudeDelta: 0.005,
            longitudeDelta: 0.005,
          }}
          showsUserLocation={true}
          userInterfaceStyle="dark"
        >
          {historicDetections.map((det) => (
            <Marker
              key={det.id}
              coordinate={{ latitude: det.latitude, longitude: det.longitude }}
              pinColor={getMarkerColor(det.severity)}
              title={`Severity: ${det.severity}`}
              description={`Confidence: ${(det.confidence * 100).toFixed(0)}%`}
            />
          ))}
        </MapView>
      ) : (
        <View style={styles.mapLoading}>
          <Text style={styles.text}>Waiting for GPS...</Text>
        </View>
      )}

      {/* Floating Control Panel */}
      <View style={styles.overlayPanel}>
        <View style={styles.statusBar}>
          <Text style={styles.title}>Model Status</Text>
          <Text style={[styles.statusText, modelLoaded ? styles.textSuccess : styles.textWarning]}>
            {modelLoaded ? 'ONNX Local - Ready' : 'Loading model...'}
          </Text>
        </View>

        <View style={styles.controlRow}>
          <Text style={styles.text}>Live Detection Running</Text>
          <Switch
            value={isActive}
            onValueChange={setIsActive}
            trackColor={{ false: Theme.colors.surfaceLight, true: Theme.colors.primary }}
            thumbColor={Theme.colors.text}
          />
        </View>

        {isActive && (
          <View style={styles.liveGauge}>
            <Text style={styles.text}>Pothole Confidence</Text>
            <View style={styles.barBackground}>
              <View 
                style={[
                  styles.barFill, 
                  { width: `${livePotholeConf * 100}%`, backgroundColor: livePotholeConf > 0.6 ? Theme.colors.secondary : Theme.colors.primary }
                ]} 
              />
            </View>
          </View>
        )}

        {lastEvent && (
          <View style={styles.eventToast}>
            <Text style={styles.eventText}>🚨 Detected: {lastEvent.className}</Text>
          </View>
        )}
      </View>
    </View>
  );
}

const styles = StyleSheet.create({
  container: {
    flex: 1,
    backgroundColor: Theme.colors.background,
  },
  map: {
    width: '100%',
    height: '100%',
  },
  mapLoading: {
    flex: 1,
    alignItems: 'center',
    justifyContent: 'center',
  },
  overlayPanel: {
    position: 'absolute',
    bottom: 20,
    alignSelf: 'center',
    width: width * 0.9,
    backgroundColor: 'rgba(20, 20, 31, 0.9)',
    borderRadius: Theme.borderRadius.lg,
    padding: Theme.spacing.md,
    borderColor: Theme.colors.surfaceLight,
    borderWidth: 1,
  },
  statusBar: {
    marginBottom: Theme.spacing.md,
  },
  title: {
    color: Theme.colors.text,
    fontSize: Theme.typography.sizes.h3,
    fontWeight: '700',
  },
  statusText: {
    fontSize: Theme.typography.sizes.body,
    marginTop: Theme.spacing.xs,
  },
  textSuccess: {
    color: Theme.colors.success,
  },
  textWarning: {
    color: Theme.colors.warning,
  },
  controlRow: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'center',
    marginVertical: Theme.spacing.sm,
  },
  text: {
    color: Theme.colors.text,
    fontSize: Theme.typography.sizes.body,
  },
  liveGauge: {
    marginTop: Theme.spacing.sm,
  },
  barBackground: {
    height: 10,
    backgroundColor: Theme.colors.surfaceLight,
    borderRadius: Theme.borderRadius.round,
    marginTop: Theme.spacing.sm,
    overflow: 'hidden',
  },
  barFill: {
    height: '100%',
    borderRadius: Theme.borderRadius.round,
  },
  eventToast: {
    marginTop: Theme.spacing.md,
    backgroundColor: 'rgba(255, 51, 102, 0.2)',
    padding: Theme.spacing.sm,
    borderRadius: Theme.borderRadius.md,
  },
  eventText: {
    color: Theme.colors.secondary,
    fontWeight: 'bold',
    textAlign: 'center',
  }
});
