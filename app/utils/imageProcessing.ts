import * as tf from '@tensorflow/tfjs';
import { classifyPatterns, analyzeFeatures, PatternPrediction } from './mlModel';

export async function preprocessImage(imageData: string): Promise<tf.Tensor4D> {
  // Create an HTML Image element
  const img = new Image();
  img.src = imageData;
  
  // Wait for image to load
  await new Promise((resolve) => {
    img.onload = resolve;
  });

  // Convert image to tensor
  const tensor = tf.browser.fromPixels(img)
    // Resize to expected size
    .resizeBilinear([224, 224])
    // Normalize pixel values
    .toFloat()
    .div(255.0)
    // Expand dimensions to create batch of 1
    .expandDims(0);

  return tensor as tf.Tensor4D;
}

export async function detectChartPatterns(tensor: tf.Tensor4D) {
  const edges = tf.tidy(() => {
    // Remove the batch dimension and get the 3D tensor
    const tensor3D = tensor.squeeze([0]) as tf.Tensor3D;
    // Convert to grayscale
    const grayscale = tf.mean(tensor3D, -1) as tf.Tensor2D;
    const sobelX = tf.tensor2d([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]]);
    const sobelY = tf.tensor2d([[-1, -2, -1], [0, 0, 0], [1, 2, 1]]);
    
    // Add channel dimension and batch dimension for conv2d
    const grayscale4D = grayscale.expandDims(-1).expandDims(0) as tf.Tensor4D;
    
    const edgesX = tf.conv2d(
      grayscale4D,
      sobelX.expandDims(-1).expandDims(-1) as tf.Tensor4D,
      1,
      'valid'
    );
    
    const edgesY = tf.conv2d(
      grayscale4D,
      sobelY.expandDims(-1).expandDims(-1) as tf.Tensor4D,
      1,
      'valid'
    );
    
    return tf.sqrt(tf.add(tf.square(edgesX), tf.square(edgesY)));
  });

  const edgeData = await edges.data();
  edges.dispose();

  // Get ML-based pattern predictions
  const mlPatterns = await classifyPatterns(tensor);
  const featureAnalysis = await analyzeFeatures(tensor);

  // Convert to Float32Array explicitly
  const edgeDataFloat32 = new Float32Array(edgeData);
  
  const patterns = await analyzePatterns(edgeDataFloat32, mlPatterns, featureAnalysis);
  return patterns;
}

async function analyzePatterns(
  edgeData: Float32Array,
  mlPatterns: PatternPrediction[],
  featureAnalysis: { complexity: number; strength: number; reliability: number }
) {
  const { trendStrength, trendDirection, volumeProfile } = detectTrend(edgeData);
  const supportResistance = findSupportResistanceLevels(edgeData);
  const traditionalPatterns = detectCandlestickPatterns(edgeData);
  const volumeAnalysis = analyzeVolume(edgeData);
  
  // Combine ML and traditional patterns
  const combinedPatterns = [
    ...traditionalPatterns,
    ...mlPatterns.map(p => p.pattern)
  ];

  // Enhanced prediction using ML features
  const prediction = predictPriceMovement(
    edgeData,
    trendStrength,
    trendDirection,
    volumeProfile,
    featureAnalysis
  );

  const indicators = calculateTechnicalIndicators(edgeData);
  
  return {
    trend: {
      strength: trendStrength * 100,
      direction: trendDirection,
      confidence: calculateConfidence(
        trendStrength,
        combinedPatterns.length,
        volumeProfile,
        featureAnalysis
      )
    },
    patterns: combinedPatterns,
    mlAnalysis: {
      patternConfidence: mlPatterns.map(p => ({
        pattern: p.pattern,
        confidence: p.confidence
      })),
      complexity: featureAnalysis.complexity,
      strength: featureAnalysis.strength,
      reliability: featureAnalysis.reliability
    },
    support: supportResistance.support,
    resistance: supportResistance.resistance,
    volume: volumeAnalysis,
    prediction: prediction,
    indicators: indicators
  };
}

function detectTrend(data: Float32Array) {
  const windowSize = 20;
  let upCount = 0;
  let downCount = 0;
  let volumeProfile = 0;

  for (let i = windowSize; i < data.length; i++) {
    const prevAvg = data.slice(i - windowSize, i).reduce((a, b) => a + b) / windowSize;
    const currentAvg = data.slice(i - windowSize + 1, i + 1).reduce((a, b) => a + b) / windowSize;
    
    if (currentAvg > prevAvg) {
      upCount++;
      volumeProfile += data[i];
    }
    if (currentAvg < prevAvg) {
      downCount++;
      volumeProfile -= data[i];
    }
  }

  const totalCount = upCount + downCount;
  const trendStrength = Math.max(upCount, downCount) / totalCount;
  const trendDirection = upCount > downCount ? 'bullish' : 'bearish';
  volumeProfile = volumeProfile / totalCount;

  return { trendStrength, trendDirection, volumeProfile };
}

function findSupportResistanceLevels(data: Float32Array) {
  const levels = new Map<number, number>();
  const threshold = 0.3;
  
  for (let i = 0; i < data.length; i++) {
    if (data[i] > threshold) {
      const level = Math.round(i / 10) * 10;
      levels.set(level, (levels.get(level) || 0) + 1);
    }
  }

  const sortedLevels = Array.from(levels.entries())
    .sort((a, b) => b[1] - a[1])
    .slice(0, 3)
    .map(([level]) => level);

  return {
    support: sortedLevels[0],
    resistance: sortedLevels[sortedLevels.length - 1]
  };
}

function detectCandlestickPatterns(data: Float32Array) {
  const patterns = [];
  const threshold = 0.4;
  
  // Basic patterns
  if (Math.abs(data[data.length - 1] - data[data.length - 2]) < threshold) {
    patterns.push('Doji');
  }
  
  if (Math.abs(data[data.length - 1] - data[data.length - 4]) > threshold * 2) {
    patterns.push('Hammer');
  }
  
  const currentBody = Math.abs(data[data.length - 1] - data[data.length - 2]);
  const previousBody = Math.abs(data[data.length - 3] - data[data.length - 4]);
  
  // Advanced patterns
  if (currentBody > previousBody * 1.5) {
    patterns.push('Engulfing');
  }

  // Morning Star pattern
  const thirdBody = Math.abs(data[data.length - 5] - data[data.length - 6]);
  if (previousBody < threshold && currentBody > threshold * 1.5 && thirdBody > threshold) {
    patterns.push('Morning Star');
  }

  // Evening Star pattern
  if (previousBody < threshold && currentBody < -threshold * 1.5 && thirdBody < -threshold) {
    patterns.push('Evening Star');
  }

  // Three White Soldiers
  if (currentBody > 0 && previousBody > 0 && thirdBody > 0 &&
      currentBody > threshold && previousBody > threshold && thirdBody > threshold) {
    patterns.push('Three White Soldiers');
  }

  // Three Black Crows
  if (currentBody < 0 && previousBody < 0 && thirdBody < 0 &&
      Math.abs(currentBody) > threshold && Math.abs(previousBody) > threshold && Math.abs(thirdBody) > threshold) {
    patterns.push('Three Black Crows');
  }

  return patterns;
}

function analyzeVolume(data: Float32Array) {
  const volumeData = data.slice(-50);  // Last 50 data points for volume analysis
  const avgVolume = volumeData.reduce((a, b) => a + b, 0) / volumeData.length;
  
  // Convert to regular array before using spread operator
  const volumeArray = Array.from(volumeData);
  const maxVolume = Math.max(...volumeArray);
  const minVolume = Math.min(...volumeArray);
  
  const volumeChange = (volumeData[volumeData.length - 1] - volumeData[0]) / volumeData[0];
  const volumeTrend = volumeChange > 0.1 ? 'increasing' : volumeChange < -0.1 ? 'decreasing' : 'stable';
  
  return {
    average: avgVolume,
    trend: volumeTrend,
    strength: (maxVolume - minVolume) / avgVolume,
    current: volumeData[volumeData.length - 1]
  };
}

function predictPriceMovement(
  data: Float32Array,
  trendStrength: number,
  trendDirection: string,
  volumeProfile: number,
  featureAnalysis: { complexity: number; strength: number; reliability: number }
) {
  const momentum = calculateMomentum(data);
  const volatility = calculateVolatility(data);
  const volumeImpact = calculateVolumeImpact(volumeProfile);
  
  // Include ML feature analysis in prediction
  const patternStrengthImpact = featureAnalysis.strength / 100;
  const reliabilityFactor = featureAnalysis.reliability / 100;
  
  const baseChange = trendStrength * (trendDirection === 'bullish' ? 1 : -1);
  const momentumFactor = 1 + momentum;
  const volatilityAdjustment = 1 - volatility * 0.5;
  const volumeAdjustment = 1 + volumeImpact;
  const mlAdjustment = 1 + (patternStrengthImpact * reliabilityFactor);
  
  const adjustedChange = baseChange * momentumFactor * volatilityAdjustment * volumeAdjustment * mlAdjustment;
  
  const confidence = calculatePredictionConfidence(
    trendStrength,
    momentum,
    volatility,
    volumeImpact,
    featureAnalysis
  );
  
  return {
    direction: adjustedChange > 0 ? 'up' : 'down',
    probability: confidence,
    expectedChange: adjustedChange * 5 // Percentage
  };
}

function calculateVolumeImpact(volumeProfile: number): number {
  return Math.tanh(volumeProfile) * 0.5;  // Normalized volume impact
}

function calculatePredictionConfidence(
  trendStrength: number,
  momentum: number,
  volatility: number,
  volumeImpact: number,
  featureAnalysis: { complexity: number; strength: number; reliability: number }
): number {
  const baseConfidence = trendStrength * 30;
  const momentumBonus = Math.abs(momentum) * 15;
  const volatilityPenalty = volatility * 10;
  const volumeBonus = Math.abs(volumeImpact) * 20;
  const mlBonus = (featureAnalysis.strength * 0.15) + (featureAnalysis.reliability * 0.10);
  
  return Math.min(
    Math.max(
      baseConfidence + momentumBonus - volatilityPenalty + volumeBonus + mlBonus,
      0
    ),
    100
  );
}

function calculateTechnicalIndicators(data: Float32Array) {
  return {
    rsi: calculateRSI(data),
    macd: calculateMACD(data),
    bollingerBands: calculateBollingerBands(data),
    stochastic: calculateStochastic(data),
    atr: calculateATR(data),
    obv: calculateOBV(data)
  };
}

function calculateRSI(data: Float32Array, period = 14) {
  let gains = 0;
  let losses = 0;
  
  for (let i = 1; i < period + 1; i++) {
    const difference = data[data.length - i] - data[data.length - i - 1];
    if (difference > 0) gains += difference;
    else losses -= difference;
  }
  
  const averageGain = gains / period;
  const averageLoss = losses / period;
  const rs = averageGain / averageLoss;
  
  return 100 - (100 / (1 + rs));
}

function calculateMACD(data: Float32Array) {
  const ema12 = calculateEMA(data, 12);
  const ema26 = calculateEMA(data, 26);
  const macd = ema12 - ema26;
  const signal = calculateEMA(new Float32Array([macd]), 9);
  
  return {
    macd,
    signal,
    histogram: macd - signal
  };
}

function calculateBollingerBands(data: Float32Array, period = 20) {
  const sma = data.slice(-period).reduce((a, b) => a + b) / period;
  const squaredDiffs = data.slice(-period).map(x => Math.pow(x - sma, 2));
  const standardDeviation = Math.sqrt(squaredDiffs.reduce((a, b) => a + b) / period);
  
  return {
    upper: sma + standardDeviation * 2,
    middle: sma,
    lower: sma - standardDeviation * 2
  };
}

function calculateStochastic(data: Float32Array, period = 14) {
  const slice = data.slice(-period);
  const currentClose = data[data.length - 1];
  
  // Convert to regular array before using spread operator
  const sliceArray = Array.from(slice);
  const lowest = Math.min(...sliceArray);
  const highest = Math.max(...sliceArray);
  
  return ((currentClose - lowest) / (highest - lowest)) * 100;
}

function calculateATR(data: Float32Array, period = 14) {
  let sum = 0;
  for (let i = 1; i < period + 1; i++) {
    const high = data[data.length - i];
    const low = data[data.length - i - 1];
    const close = data[data.length - i - 2];
    
    const tr = Math.max(
      high - low,
      Math.abs(high - close),
      Math.abs(low - close)
    );
    sum += tr;
  }
  
  return sum / period;
}

function calculateOBV(data: Float32Array) {
  let obv = 0;
  for (let i = 1; i < data.length; i++) {
    if (data[i] > data[i - 1]) {
      obv += data[i];
    } else if (data[i] < data[i - 1]) {
      obv -= data[i];
    }
  }
  return obv;
}

function calculateMomentum(data: Float32Array, period = 10) {
  const current = data[data.length - 1];
  const previous = data[data.length - period];
  return (current - previous) / previous;
}

function calculateVolatility(data: Float32Array, period = 10) {
  const prices = data.slice(-period);
  const mean = prices.reduce((a, b) => a + b) / period;
  const squaredDiffs = prices.map(x => Math.pow(x - mean, 2));
  return Math.sqrt(squaredDiffs.reduce((a, b) => a + b) / period);
}

function calculateEMA(data: Float32Array, period: number) {
  const multiplier = 2 / (period + 1);
  let ema = data[0];
  
  for (let i = 1; i < data.length; i++) {
    ema = (data[i] - ema) * multiplier + ema;
  }
  
  return ema;
}

function calculateConfidence(
  trendStrength: number,
  patternCount: number,
  volumeProfile: number,
  featureAnalysis: { complexity: number; strength: number; reliability: number }
) {
  const baseConfidence = trendStrength * 40;
  const patternBonus = Math.min(patternCount * 8, 25);
  const volumeConfidence = Math.abs(volumeProfile) * 15;
  const mlConfidence = (featureAnalysis.strength * 0.1) + (featureAnalysis.reliability * 0.1);
  
  return Math.min(baseConfidence + patternBonus + volumeConfidence + mlConfidence, 100);
}