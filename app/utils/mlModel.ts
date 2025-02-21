import * as tf from '@tensorflow/tfjs';
import * as mobilenet from '@tensorflow-models/mobilenet';

// Define the model architecture for pattern recognition
export async function createPatternRecognitionModel() {
  const model = tf.sequential();

  model.add(tf.layers.conv2d({
    inputShape: [224, 224, 1],
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

  model.add(tf.layers.flatten({ name: 'flatten' }));
  model.add(tf.layers.dense({
    units: 256,
    activation: 'relu',
    kernelRegularizer: tf.regularizers.l2({ l2: 0.001 }),
    name: 'features'
  }));
  model.add(tf.layers.dropout({ rate: 0.5, name: 'dropout' }));
  model.add(tf.layers.dense({
    units: 8,
    activation: 'softmax',
    name: 'output'
  }));

  model.compile({
    optimizer: tf.train.adam(0.0001),
    loss: 'categoricalCrossentropy',
    metrics: ['accuracy']
  });

  return model;
}

// Transfer Learning Integration
export async function createTransferLearningModel() {
  const baseModel = await mobilenet.load();
  const mobileNetLayers = baseModel.model.layers;

  const featureExtractor = tf.model({
    inputs: mobileNetLayers[0].input,
    outputs: mobileNetLayers[mobileNetLayers.length - 2].output
  });

  featureExtractor.trainable = false;

  const flattenLayer = tf.layers.flatten();
  const denseLayer = tf.layers.dense({
    units: 256,
    activation: 'relu',
    kernelRegularizer: tf.regularizers.l2({ l2: 0.001 })
  });
  const dropoutLayer = tf.layers.dropout({ rate: 0.5 });
  const outputLayer = tf.layers.dense({
    units: 8,
    activation: 'softmax'
  });

  const model = tf.sequential();
  model.add(featureExtractor);
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

// Helper functions for preprocessing
function preprocessTensor(tensor: tf.Tensor4D): tf.Tensor4D {
  return tf.tidy(() => {
    const tensor3D = tensor.squeeze([0]) as tf.Tensor3D;
    const grayscale = tf.mean(tensor3D, -1).expandDims(-1);
    const rgb = grayscale.tile([1, 1, 3]); // Convert to RGB
    return rgb.div(tf.scalar(255.0)).expandDims(0);
  });
}

// Ensemble Method
export async function ensemblePredictions(tensors: tf.Tensor4D[]): Promise<number[]> {
  const models = [await createPatternRecognitionModel(), await createTransferLearningModel()];

  const predictions = await Promise.all(
    models.map(model =>
      tf.tidy(() => {
        const processedTensors = tensors.map(tensor => preprocessTensor(tensor));
        return model.predict(tf.concat(processedTensors, 0))?.dataSync();
      })
    )
  );

  const averagedPredictions = predictions[0].map((value, index) =>
    (value + predictions[1][index]) / 2
  );

  models.forEach(model => model.dispose());
  return averagedPredictions;
}
