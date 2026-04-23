import React from 'react';
import { StyleSheet, View, Text, Switch } from 'react-native';
import { Theme } from '../../constants/theme';

export default function SettingsScreen() {
  const [backgroundMode, setBackgroundMode] = React.useState(false);
  const [hapticFeedback, setHapticFeedback] = React.useState(true);
  
  return (
    <View style={styles.container}>
      <Text style={styles.title}>Settings</Text>

      <View style={styles.settingCard}>
        <View style={styles.settingRow}>
          <View>
            <Text style={styles.settingText}>Background Tracking</Text>
            <Text style={styles.textMuted}>Log potholes while app is minimized</Text>
          </View>
          <Switch 
            value={backgroundMode} 
            onValueChange={setBackgroundMode}
            trackColor={{ false: Theme.colors.surfaceLight, true: Theme.colors.primary }}
            thumbColor={Theme.colors.text}
          />
        </View>

        <View style={styles.divider} />

        <View style={styles.settingRow}>
          <View>
            <Text style={styles.settingText}>Haptic Alerts</Text>
            <Text style={styles.textMuted}>Vibrate when pothole detected</Text>
          </View>
          <Switch 
            value={hapticFeedback} 
            onValueChange={setHapticFeedback}
            trackColor={{ false: Theme.colors.surfaceLight, true: Theme.colors.primary }}
            thumbColor={Theme.colors.text}
          />
        </View>
      </View>

      <Text style={styles.infoText}>PaveSense v1.0 - On-Device ML MVP</Text>
    </View>
  );
}

const styles = StyleSheet.create({
  container: {
    flex: 1,
    backgroundColor: Theme.colors.background,
    padding: Theme.spacing.md,
  },
  title: {
    color: Theme.colors.text,
    fontSize: Theme.typography.sizes.h2,
    fontWeight: 'bold',
    marginBottom: Theme.spacing.md,
  },
  settingCard: {
    backgroundColor: Theme.colors.surface,
    borderRadius: Theme.borderRadius.md,
    padding: Theme.spacing.md,
    borderWidth: 1,
    borderColor: Theme.colors.surfaceLight,
  },
  settingRow: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'center',
    paddingVertical: Theme.spacing.sm,
  },
  settingText: {
    color: Theme.colors.text,
    fontSize: Theme.typography.sizes.body,
    fontWeight: '500',
  },
  textMuted: {
    color: Theme.colors.textMuted,
    fontSize: Theme.typography.sizes.small,
    marginTop: 2,
  },
  divider: {
    height: 1,
    backgroundColor: Theme.colors.surfaceLight,
    marginVertical: Theme.spacing.sm,
  },
  infoText: {
    color: Theme.colors.textMuted,
    textAlign: 'center',
    marginTop: Theme.spacing.xl,
  }
});
