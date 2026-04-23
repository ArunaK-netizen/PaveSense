import React, { useEffect, useState } from 'react';
import { StyleSheet, View, Text, ActivityIndicator, TouchableOpacity, Dimensions } from 'react-native';
import * as Updates from 'expo-updates';
import { Theme } from '../constants/theme';

const { width } = Dimensions.get('window');

/**
 * Custom OTA screen wrapping main content.
 * Checks for JS updates securely and downloads them over-the-air.
 */
export const OtaUpdater = ({ children }: { children: React.ReactNode }) => {
  const { currentlyRunning, isUpdateAvailable, isUpdatePending, downloadedUpdate } = Updates.useUpdates();
  const [downloading, setDownloading] = useState(false);

  useEffect(() => {
    // If update available right away and we're not downloading, trigger it automatically.
    if (isUpdateAvailable && !downloading && !isUpdatePending) {
      handleDownload();
    }
  }, [isUpdateAvailable]);

  const handleDownload = async () => {
    setDownloading(true);
    try {
      await Updates.fetchUpdateAsync();
    } catch (e) {
      console.error('Failed to download update:', e);
      setDownloading(false);
    }
  };

  const handleRestart = async () => {
    await Updates.reloadAsync();
  };

  const showUpdateScreen = isUpdateAvailable || downloading || isUpdatePending;

  if (showUpdateScreen) {
    return (
      <View style={styles.container}>
        <View style={styles.card}>
          <Text style={styles.title}>PaveSense Update</Text>
          
          {isUpdatePending ? (
            <>
              <Text style={styles.subtitle}>Update is ready! ✨</Text>
              <Text style={styles.versionInfo}>
                New Update: {downloadedUpdate?.updateId || 'Latest'}
              </Text>
              <TouchableOpacity style={styles.button} onPress={handleRestart}>
                <Text style={styles.buttonText}>Restart App Now</Text>
              </TouchableOpacity>
            </>
          ) : (
            <>
              <Text style={styles.subtitle}>Downloading new patch...</Text>
              <ActivityIndicator size="large" color={Theme.colors.primary} style={styles.loader} />
              <Text style={styles.versionInfo}>
                Current: {currentlyRunning.updateId || 'Base'}
              </Text>
            </>
          )}
        </View>
      </View>
    );
  }

  return <>{children}</>;
};

const styles = StyleSheet.create({
  container: {
    flex: 1,
    backgroundColor: Theme.colors.background,
    justifyContent: 'center',
    alignItems: 'center',
  },
  card: {
    width: width * 0.85,
    backgroundColor: Theme.colors.surface,
    borderRadius: Theme.borderRadius.lg,
    padding: Theme.spacing.xl,
    alignItems: 'center',
    borderWidth: 1,
    borderColor: Theme.colors.surfaceLight,
  },
  title: {
    color: Theme.colors.text,
    fontSize: Theme.typography.sizes.h2,
    fontWeight: 'bold',
    marginBottom: Theme.spacing.sm,
  },
  subtitle: {
    color: Theme.colors.textMuted,
    fontSize: Theme.typography.sizes.body,
    textAlign: 'center',
    marginBottom: Theme.spacing.lg,
  },
  versionInfo: {
    color: Theme.colors.textMuted,
    fontSize: Theme.typography.sizes.small,
    marginTop: Theme.spacing.lg,
    marginBottom: Theme.spacing.md,
    fontFamily: 'monospace',
  },
  loader: {
    marginVertical: Theme.spacing.md,
  },
  button: {
    backgroundColor: Theme.colors.primary,
    paddingVertical: Theme.spacing.md,
    paddingHorizontal: Theme.spacing.xl,
    borderRadius: Theme.borderRadius.md,
    width: '100%',
  },
  buttonText: {
    color: Theme.colors.background,
    fontWeight: 'bold',
    fontSize: Theme.typography.sizes.body,
    textAlign: 'center',
  }
});
