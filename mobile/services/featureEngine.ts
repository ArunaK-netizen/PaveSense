export const FEATURE_MEAN = [
  -0.0051102167926728725, -0.05571748688817024, 9.240255355834961,
  0.016977300867438316, 0.0014803556259721518, -0.0020655214320868254,
  9.536758422851562, 0.2223033607006073, -0.15091516077518463,
  -0.10761234909296036, 0.008308748714625835, 0.02501893788576126,
  0.043939560651779175
];

export const FEATURE_VAR = [
  4.073901653289795, 3.9377474784851074, 6.8808979988098145,
  0.40692946314811707, 0.373217910528183, 0.3899006247520447,
  8.874469757080078, 1.0960880517959595, 16200.8671875,
  6.169461250305176, 0.2318536341190338, 0.01478387787938118,
  0.02327854186296463
];

const GRAVITY = 9.81;
const SAMPLING_RATE = 50;
const POTHOLE_FREQ_LOW = 5.0;
const POTHOLE_FREQ_HIGH = 25.0;

export function extractFeatures(
  accel: {x: number[], y: number[], z: number[]},
  gyro: {x: number[], y: number[], z: number[]}
): number[][] {
  const seqLen = accel.x.length;
  const features: number[][] = Array(seqLen).fill(0).map(() => Array(13).fill(0));

  let gravityEstimate = [...accel.z].sort()[Math.floor(seqLen / 2)] || GRAVITY;
  
  for (let i = 0; i < seqLen; i++) {
    // 0-5: Raw
    features[i][0] = accel.x[i];
    features[i][1] = accel.y[i];
    features[i][2] = accel.z[i];
    features[i][3] = gyro.x[i];
    features[i][4] = gyro.y[i];
    features[i][5] = gyro.z[i];

    // 6-7: Magnitudes
    const aMag = Math.sqrt(accel.x[i]**2 + accel.y[i]**2 + accel.z[i]**2);
    features[i][6] = aMag;
    features[i][7] = Math.sqrt(gyro.x[i]**2 + gyro.y[i]**2 + gyro.z[i]**2);

    // 8: Jerk Z
    if (i === 0) features[i][8] = 0;
    else features[i][8] = (accel.z[i] - accel.z[i-1]) * SAMPLING_RATE;

    // 9: Detrended Z
    features[i][9] = accel.z[i] - gravityEstimate;

    // 10: Vertical asymmetry
    features[i][10] = computeAsymmetry(accel.z.map(z => z - gravityEstimate), i, 10);

    // 11: Spectral Energy (simplified for JS — moving variance in short window as proxy)
    features[i][11] = computeProxyEnergy(accel.z, i, 20);

    // 12: Freefall Score
    features[i][12] = Math.max(0, Math.min(1, 1.0 - (aMag / GRAVITY)));

    // Normalize
    for (let f = 0; f < 13; f++) {
      features[i][f] = (features[i][f] - FEATURE_MEAN[f]) / Math.sqrt(FEATURE_VAR[f]);
    }
  }

  // padding jerk_z
  if (seqLen > 0) features[0][8] = features.length > 1 ? features[1][8] : 0;

  return features;
}

function computeAsymmetry(detrendedZ: number[], idx: number, window: number) {
  const half = Math.floor(window / 2);
  if (idx < half || idx >= detrendedZ.length - half) return 0;
  
  let posPeak = 0;
  let negPeak = 0;
  for (let j = idx - half; j < idx + half; j++) {
    if (detrendedZ[j] > posPeak) posPeak = detrendedZ[j];
    if (Math.abs(detrendedZ[j]) > negPeak && detrendedZ[j] < 0) negPeak = Math.abs(detrendedZ[j]);
  }
  
  const denom = posPeak + negPeak;
  if (denom > 0.5) return (negPeak - posPeak) / denom;
  return 0;
}

// Proxies frequency energy using local variance, as JS doesn't have an easy FFT
function computeProxyEnergy(z: number[], idx: number, window: number) {
  const half = Math.floor(window / 2);
  if (idx < half || idx >= z.length - half) return 0;
  
  let sum = 0;
  for (let j = idx - half; j < idx + half; j++) {
    sum += z[j];
  }
  const mean = sum / window;
  let sqSum = 0;
  for (let j = idx - half; j < idx + half; j++) {
    sqSum += Math.pow(z[j] - mean, 2);
  }
  return sqSum / window;
}
