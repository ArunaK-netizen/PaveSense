import { InferenceSession, Tensor } from 'onnxruntime-react-native';
import { Asset } from 'expo-asset';

export const EVENT_CLASSES = [
  'normal',
  'pothole',
  'speed_bump',
  'phone_drop',
  'disturbance'
];

export class PaveSenseML {
  private session: InferenceSession | null = null;
  private isLoaded = false;

  async init() {
    try {
      console.log('Loading ML model...');
      // Ensure Metro resolves the `.onnx` extension properly through metro.config.js
      const [{ localUri }] = await Asset.loadAsync(require('../assets/pavesense_advanced.onnx'));
      
      if (!localUri) throw new Error('Failed to load ONNX model uri');

      this.session = await InferenceSession.create(localUri);
      this.isLoaded = true;
      console.log('✅ ML Module loaded');
    } catch (error) {
      console.error('❌ Failed to load ML model:', error);
    }
  }

  async predict(features: number[][]) {
    if (!this.isLoaded || !this.session) return null;

    // features is 100x13
    // Flatten the array to pass it to tensor
    const flattened = features.flat();
    const inputDims = [1, 100, 13];

    // Note: Float32Array
    const inputTensor = new Tensor('float32', new Float32Array(flattened), inputDims);

    try {
      // The input name in the ONNX export script was 'sensor_input'
      const feeds: Record<string, Tensor> = {};
      feeds['sensor_input'] = inputTensor;

      const outputData = await this.session.run(feeds);
      const logits = outputData['class_logits'].data as Float32Array;

      // Softmax over logits
      const probs = this.softmax(logits);
      
      // Determine class
      let maxIdx = 0;
      let maxProb = probs[0];
      for (let i = 1; i < 5; i++) {
        if (probs[i] > maxProb) {
          maxProb = probs[i];
          maxIdx = i;
        }
      }

      return {
        classIdx: maxIdx,
        className: EVENT_CLASSES[maxIdx],
        confidence: maxProb,
        potholeConfidence: probs[1],
        probabilities: Array.from(probs)
      };

    } catch (e) {
      console.error('Error during prediction:', e);
      return null;
    }
  }

  private softmax(arr: Float32Array) {
    const max = Math.max(...arr);
    const exps = arr.map(e => Math.exp(e - max));
    const sum = exps.reduce((a, b) => a + b, 0);
    return exps.map(e => e / sum);
  }
}
