import React, { useEffect, useState } from 'react';
import { StyleSheet, View, Text, FlatList, TouchableOpacity } from 'react-native';
import { Theme } from '../../constants/theme';
import { getDetections, clearDetections } from '../../services/database';

export default function HistoryScreen() {
  const [history, setHistory] = useState<any[]>([]);

  useEffect(() => {
    loadDetections();
  }, []);

  const loadDetections = async () => {
    const data = await getDetections();
    setHistory(data);
  };

  const handleClear = async () => {
    await clearDetections();
    setHistory([]);
  };

  const renderItem = ({ item }: { item: any }) => (
    <View style={styles.historyCard}>
      <View style={styles.cardHeader}>
        <Text style={styles.eventClass}>{item.category || 'Pothole'}</Text>
        <View style={[styles.severityBadge, { backgroundColor: item.severity === 'high' ? Theme.colors.markerHigh : Theme.colors.markerMedium }]}>
          <Text style={styles.severityText}>{item.severity}</Text>
        </View>
      </View>
      
      <Text style={styles.textMuted}>Confidence: {(item.confidence * 100).toFixed(1)}%</Text>
      <Text style={styles.textMuted}>Location: {item.latitude.toFixed(5)}, {item.longitude.toFixed(5)}</Text>
      <Text style={styles.textMuted}>Date: {new Date(item.timestamp).toLocaleString()}</Text>
    </View>
  );

  return (
    <View style={styles.container}>
      <View style={styles.headerRow}>
        <Text style={styles.title}>Detection History</Text>
        <TouchableOpacity style={styles.clearButton} onPress={handleClear}>
          <Text style={styles.clearText}>Clear</Text>
        </TouchableOpacity>
      </View>
      
      {history.length === 0 ? (
        <View style={styles.emptyContainer}>
           <Text style={styles.textMuted}>No detections logged yet.</Text>
        </View>
      ) : (
        <FlatList
          data={history}
          keyExtractor={(item) => String(item.id)}
          renderItem={renderItem}
          contentContainerStyle={{ paddingBottom: 20 }}
        />
      )}
    </View>
  );
}

const styles = StyleSheet.create({
  container: {
    flex: 1,
    backgroundColor: Theme.colors.background,
    padding: Theme.spacing.md,
  },
  headerRow: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'center',
    marginBottom: Theme.spacing.md,
  },
  title: {
    color: Theme.colors.text,
    fontSize: Theme.typography.sizes.h2,
    fontWeight: 'bold',
  },
  clearButton: {
    backgroundColor: Theme.colors.surfaceLight,
    paddingVertical: Theme.spacing.xs,
    paddingHorizontal: Theme.spacing.sm,
    borderRadius: Theme.borderRadius.md,
  },
  clearText: {
    color: Theme.colors.secondary,
  },
  emptyContainer: {
    flex: 1,
    justifyContent: 'center',
    alignItems: 'center',
  },
  historyCard: {
    backgroundColor: Theme.colors.surface,
    padding: Theme.spacing.md,
    borderRadius: Theme.borderRadius.md,
    marginBottom: Theme.spacing.sm,
    borderWidth: 1,
    borderColor: Theme.colors.surfaceLight,
  },
  cardHeader: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    marginBottom: Theme.spacing.xs,
  },
  eventClass: {
    color: Theme.colors.text,
    fontSize: Theme.typography.sizes.h3,
    fontWeight: 'bold',
    textTransform: 'capitalize',
  },
  severityBadge: {
    paddingHorizontal: 8,
    paddingVertical: 4,
    borderRadius: Theme.borderRadius.round,
  },
  severityText: {
    color: '#FFF',
    fontSize: 12,
    fontWeight: 'bold',
    textTransform: 'uppercase',
  },
  textMuted: {
    color: Theme.colors.textMuted,
    fontSize: Theme.typography.sizes.body,
    marginTop: 2,
  }
});
