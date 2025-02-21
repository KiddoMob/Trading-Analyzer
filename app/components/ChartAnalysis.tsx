'use client';

import { Line } from 'react-chartjs-2';
import {
  Chart as ChartJS,
  CategoryScale,
  LinearScale,
  PointElement,
  LineElement,
  Title,
  Tooltip,
  Legend
} from 'chart.js';

ChartJS.register(
  CategoryScale,
  LinearScale,
  PointElement,
  LineElement,
  Title,
  Tooltip,
  Legend
);

interface MLAnalysis {
  patternConfidence: Array<{
    pattern: string;
    confidence: number;
  }>;
  complexity: number;
  strength: number;
  reliability: number;
}

interface PredictionType {
  trend: {
    strength: number;
    direction: string;
    confidence: number;
  };
  patterns: string[];
  mlAnalysis: MLAnalysis;
  support: number;
  resistance: number;
  volume: {
    average: number;
    trend: string;
    strength: number;
    current: number;
  };
  prediction: {
    direction: string;
    probability: number;
    expectedChange: number;
  };
  indicators: {
    rsi: number;
    macd: {
      macd: number;
      signal: number;
      histogram: number;
    };
    bollingerBands: {
      upper: number;
      middle: number;
      lower: number;
    };
    stochastic: number;
    atr: number;
    obv: number;
  };
}

interface ChartAnalysisProps {
  prediction: PredictionType | null;
}

export default function ChartAnalysis({ prediction }: ChartAnalysisProps) {
  if (!prediction) return null;

  const data = {
    labels: ['Current', '1h', '4h', '1d', '1w'],
    datasets: [
      {
        label: 'Predicted Price Movement',
        data: [100, 
          100 * (1 + prediction.prediction.expectedChange / 100),
          100 * (1 + prediction.prediction.expectedChange / 100 * 1.5),
          100 * (1 + prediction.prediction.expectedChange / 100 * 2),
          100 * (1 + prediction.prediction.expectedChange / 100 * 2.5)
        ],
        borderColor: prediction.prediction.direction === 'up' ? 'rgb(75, 192, 92)' : 'rgb(255, 99, 132)',
        tension: 0.1
      },
      {
        label: 'Upper Bollinger Band',
        data: Array(5).fill(prediction.indicators.bollingerBands.upper * 100),
        borderColor: 'rgba(75, 192, 192, 0.5)',
        borderDash: [5, 5],
        tension: 0.1
      },
      {
        label: 'Lower Bollinger Band',
        data: Array(5).fill(prediction.indicators.bollingerBands.lower * 100),
        borderColor: 'rgba(75, 192, 192, 0.5)',
        borderDash: [5, 5],
        tension: 0.1
      }
    ]
  };

  const options = {
    responsive: true,
    plugins: {
      legend: {
        position: 'top' as const,
      },
      title: {
        display: true,
        text: 'Price Prediction & Technical Indicators'
      }
    }
  };

  return (
    <div className="mt-4">
      <div className="bg-white p-4 rounded-lg shadow">
        <h3 className="text-xl font-semibold mb-4">Technical Analysis</h3>
        
        <div className="grid grid-cols-2 gap-4 mb-4">
          <div className="p-3 bg-gray-50 rounded">
            <h4 className="font-semibold mb-2">Trend Analysis</h4>
            <p className="text-gray-700">
              <strong>Direction:</strong> {prediction.trend.direction}
            </p>
            <p className="text-gray-700">
              <strong>Strength:</strong> {prediction.trend.strength.toFixed(2)}%
            </p>
            <p className="text-gray-700">
              <strong>Confidence:</strong> {prediction.trend.confidence.toFixed(2)}%
            </p>
          </div>
          
          <div className="p-3 bg-gray-50 rounded">
            <h4 className="font-semibold mb-2">ML Analysis</h4>
            <p className="text-gray-700">
              <strong>Pattern Strength:</strong> {prediction.mlAnalysis.strength.toFixed(2)}%
            </p>
            <p className="text-gray-700">
              <strong>Reliability:</strong> {prediction.mlAnalysis.reliability.toFixed(2)}%
            </p>
            <p className="text-gray-700">
              <strong>Complexity:</strong> {prediction.mlAnalysis.complexity.toFixed(2)}%
            </p>
          </div>
        </div>

        <div className="grid grid-cols-2 gap-4 mb-4">
          <div className="p-3 bg-gray-50 rounded">
            <h4 className="font-semibold mb-2">Volume Analysis</h4>
            <p className="text-gray-700">
              <strong>Trend:</strong> {prediction.volume.trend}
            </p>
            <p className="text-gray-700">
              <strong>Strength:</strong> {prediction.volume.strength.toFixed(2)}
            </p>
            <p className="text-gray-700">
              <strong>Average:</strong> {prediction.volume.average.toFixed(2)}
            </p>
          </div>
          
          <div className="p-3 bg-gray-50 rounded">
            <h4 className="font-semibold mb-2">Price Prediction</h4>
            <p className="text-gray-700">
              <strong>Direction:</strong> {prediction.prediction.direction}
            </p>
            <p className="text-gray-700">
              <strong>Probability:</strong> {prediction.prediction.probability.toFixed(2)}%
            </p>
            <p className="text-gray-700">
              <strong>Expected Change:</strong> {prediction.prediction.expectedChange.toFixed(2)}%
            </p>
          </div>
        </div>

        <div className="grid grid-cols-2 gap-4 mb-4">
          <div className="p-3 bg-gray-50 rounded">
            <h4 className="font-semibold mb-2">Support/Resistance</h4>
            <p className="text-gray-700">
              <strong>Support:</strong> {prediction.support}
            </p>
            <p className="text-gray-700">
              <strong>Resistance:</strong> {prediction.resistance}
            </p>
          </div>
          
          <div className="p-3 bg-gray-50 rounded">
            <h4 className="font-semibold mb-2">Technical Indicators</h4>
            <p className="text-gray-700">
              <strong>RSI:</strong> {prediction.indicators.rsi.toFixed(2)}
            </p>
            <p className="text-gray-700">
              <strong>MACD:</strong> {prediction.indicators.macd.macd.toFixed(2)}
            </p>
            <p className="text-gray-700">
              <strong>Stochastic:</strong> {prediction.indicators.stochastic.toFixed(2)}
            </p>
          </div>
        </div>

        <div className="mb-4">
          <h4 className="font-semibold mb-2">ML-Detected Patterns</h4>
          <div className="flex flex-wrap gap-2">
            {prediction.mlAnalysis.patternConfidence.map((pattern, index) => (
              <span
                key={index}
                className="px-3 py-1 bg-blue-100 text-blue-800 rounded-full text-sm"
                title={`Confidence: ${pattern.confidence.toFixed(1)}%`}
              >
                {pattern.pattern}
              </span>
            ))}
          </div>
        </div>

        {prediction.patterns.length > 0 && (
          <div className="mb-4">
            <h4 className="font-semibold mb-2">Traditional Patterns</h4>
            <div className="flex flex-wrap gap-2">
              {prediction.patterns.map((pattern, index) => (
                <span key={index} className="px-3 py-1 bg-green-100 text-green-800 rounded-full text-sm"> ```
                  {pattern}
                </span>
              ))}
            </div>
          </div>
        )}

        <div className="h-64">
          <Line options={options} data={data} />
        </div>
      </div>
    </div>
  );
}