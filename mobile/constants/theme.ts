export const Theme = {
  colors: {
    background: '#0A0A0F',
    surface: '#14141F',
    surfaceLight: '#1E1E2E',
    primary: '#00F0FF',
    secondary: '#FF3366',
    success: '#00FF88',
    warning: '#FFB800',
    text: '#E8E8F0',
    textMuted: '#6B6B80',
    markerHigh: '#ff3366',
    markerMedium: '#ffb800',
    markerLow: '#00ff88',
  },
  typography: {
    fontFamily: 'System', // Use system default for MVP, can add custom fonts later
    sizes: {
      small: 12,
      body: 16,
      h3: 20,
      h2: 24,
      h1: 32,
    },
    weights: {
      normal: '400' as const,
      medium: '500' as const,
      bold: '700' as const,
    }
  },
  spacing: {
    xs: 4,
    sm: 8,
    md: 16,
    lg: 24,
    xl: 32,
  },
  borderRadius: {
    sm: 4,
    md: 8,
    lg: 16,
    round: 9999,
  }
};
