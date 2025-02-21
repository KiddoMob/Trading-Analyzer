import * as tf from '@tensorflow/tfjs';

// Define the model architecture for pattern recognition
export async function createPatternRecognitionModel() {
  const model = tf.sequential();

  // Convolutional layers for feature extraction
  model.add(tf.layers.conv2d({
    inputShape: [224, 224, 1],  // Single channel (grayscale)
    filters: 32,
    kernelSize: 3,
    activation: 'relu',
    padding: 'same',
    name: 'conv1'
  }));
  model.add(tf.layers.batchNormalization());
  model.add(tf.layers.maxPooling2d({ poolSize: 2, name: 'pool1' }));

  model.add(tf.layers.conv2d({
    filters: 64,
    kernelSize: 3,
    activation: 'relu',
    padding: 'same',
    name: 'conv2'
  }));
  model.add(tf.layers.batchNormalization());
  model.add(tf.layers.maxPooling2d({ poolSize: 2, name: 'pool2' }));

  model.add(tf.layers.conv2d({
    filters: 128,
    kernelSize: 3,
    activation: 'relu',
    padding: 'same',
    name: 'conv3'
  }));
  model.add(tf.layers.batchNormalization());
  model.add(tf.layers.maxPooling2d({ poolSize: 2, name: 'pool3' }));

  // Flatten and dense layers for classification
  model.add(tf.layers.flatten({ name: 'flatten' }));
  model.add(tf.layers.dense({
    units: 256,
    activation: 'relu',
    kernelRegularizer: tf.regularizers.l2({ l2: 0.001 }), // Corrected L2 regularization
    name: 'features'  // Named layer for feature extraction
  }));
  model.add(tf.layers.dropout({ rate: 0.5, name: 'dropout' }));
  model.add(tf.layers.dense({
    units: 8,
    activation: 'softmax', // Use softmax for single-label classification
    name: 'output'
  }));

  model.compile({
    optimizer: tf.train.adam(0.0001), // Lower learning rate
    loss: 'categoricalCrossentropy', // Use categorical cross-entropy for multi-class
    metrics: ['accuracy']
  });

  return model;
}

// Data Augmentation Function
function augmentData(tensor: tf.Tensor4D) {
  return tf.tidy(() => {
    const augmentedTensor = tensor.clone();
    const randomRotation = Math.random() < 0.5 ? tf.image.rotateWithOffset(augmentedTensor, Math.PI / 4) : augmentedTensor;
    const randomFlip = Math.random() < 0.5 ? tf.image.flipLeftRight(randomRotation) : randomRotation;
    return randomFlip;
  });
}

// Pattern recognition using ML model
export async function recognizePatterns(tensor: tf.Tensor4D) {
  const model = await createPatternRecognitionModel();

  // Preprocess the tensor
  const processedTensor = tf.tidy(() => {
    const tensor3D = tensor.squeeze([0]) as tf.Tensor3D;
    const grayscale = tf.mean(tensor3D, -1).expandDims(-1); // Convert to grayscale
    return grayscale.div(tf.scalar(255.0)).expandDims(0); // Normalize and add batch dimension
  });

  // Apply data augmentation during training
  const augmentedTensor = augmentData(processedTensor);

  // Get pattern predictions
  const predictions = await model.predict(augmentedTensor) as tf.Tensor;
  const probabilities = await predictions.data();

  // Cleanup
  processedTensor.dispose();
  augmentedTensor.dispose();
  predictions.dispose();

  return probabilities;
}

// Feature extraction using CNN
export async function extractFeatures(tensor: tf.Tensor4D) {
  const model = await createPatternRecognitionModel();

  // Get the feature extraction layers
  const featureModel = tf.model({
    inputs: model.inputs,
    outputs: model.getLayer('features').output // Use the named features layer
  });

  // Preprocess tensor similar to recognizePatterns
  const processedTensor = tf.tidy(() => {
    const tensor3D = tensor.squeeze([0]) as tf.Tensor3D;
    const grayscale = tf.mean(tensor3D, -1).expandDims(-1); // Convert to grayscale
    return grayscale.div(tf.scalar(255.0)).expandDims(0); // Normalize and add batch dimension
  });

  // Extract features
  const features = await featureModel.predict(processedTensor) as tf.Tensor;
  const featureArray = await features.data();

  // Cleanup
  processedTensor.dispose();
  features.dispose();

  return featureArray;
}

// Pattern classification with confidence scores
export interface PatternPrediction {
  pattern: string;
  confidence: number;
}
const PATTERN_NAMES = [
  'Double Top',
  'Double Bottom',
  'Head and Shoulders',
  'Inverse Head and Shoulders',
  'Triangle',
  'Flag',
  'Wedge',
  'Channel'
];
export async function classifyPatterns(tensor: tf.Tensor4D): Promise<PatternPrediction[]> {
  const probabilities = await recognizePatterns(tensor);

  // Find the index of the highest probability
  const maxIndex = probabilities.reduce((iMax, x, i, arr) => x > arr[iMax] ? i : iMax, 0);

  return [
    {
      pattern: PATTERN_NAMES[maxIndex],
      confidence: probabilities[maxIndex] * 100
    }
  ];
}

// Advanced feature analysis
export async function analyzeFeatures(tensor: tf.Tensor4D): Promise<{
  complexity: number;
  strength: number;
  reliability: number;
}> {
  const features = await extractFeatures(tensor);

  return {
    complexity: calculateComplexity(features),
    strength: calculatePatternStrength(features),
    reliability: calculateReliability(features)
  };
}

// Transfer Learning Integration
export async function createTransferLearningModel() {
  const baseModel = tf.applications.MobileNetV2({
    inputShape: [224, 224, 1],
    weights: 'imagenet',
    includeTop: false
  });

  // Freeze the base model layers
  baseModel.trainable = false;

  const flattenLayer = tf.layers.flatten();
  const denseLayer = tf.layers.dense({
    units: 256,
    activation: 'relu',
    kernelRegularizer: tf.regularizers.l2({ l2: 0.001 }) // Corrected L2 regularization
  });
  const dropoutLayer = tf.layers.dropout({ rate: 0.5 });
  const outputLayer = tf.layers.dense({
    units: 8,
    activation: 'softmax'
  });

  const model = tf.sequential();
  model.add(baseModel);
  model.add(flattenLayer);
  model.add(denseLayer);
  model.add(dropoutLayer);
  model.add(outputLayer);

  model.compile({
    optimizer: tf.train.adam(0.0001),
    loss: 'categoricalCrossentropy',
    metrics: ['accuracy']
  });

  return model;
}

// Ensemble Method
export async function ensemblePredictions(tensors: tf.Tensor4D[]): Promise<number[]> {
  const models = [await createPatternRecognitionModel(), await createTransferLearningModel()];
  const predictions = await Promise.all(
    models.map(model =>
      tf.tidy(() => model.predict(tf.concat(tensors, 0))?.data())
    )
  );

  // Average the predictions
  const averagedPredictions = predictions[0].map((value, index) =>
    (value + predictions[1][index]) / 2
  );

  models.forEach(model => model.dispose());
  return averagedPredictions;
}

// Helper functions for feature analysis
function calculateComplexity(features: Float32Array): number {
  const mean = features.reduce((a, b) => a + b, 0) / features.length;
  const variance = Array.from(features).reduce((sum, val) => sum + Math.pow(val - mean, 2), 0) / features.length;
  return Math.min(variance * 100, 100);
}

function calculatePatternStrength(features: Float32Array): number {
  const maxActivation = Math.max(...Array.from(features));
  return maxActivation * 100;
}

function calculateReliability(features: Float32Array): number {
  const mean = features.reduce((a, b) => a + b, 0) / features.length;
  const maxVal = Math.max(...Array.from(features));
  return (1 - Math.abs(mean / maxVal)) * 100;
}