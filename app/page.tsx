'use client';

import { useCallback, useState, useEffect } from 'react';
import { useDropzone } from 'react-dropzone';
import Image from 'next/image';
import { preprocessImage, detectChartPatterns } from './utils/imageProcessing';
import ChartAnalysis from './components/ChartAnalysis';

export default function Home() {
  const [image, setImage] = useState<string | null>(null);
  const [prediction, setPrediction] = useState<any>(null);
  const [isAnalyzing, setIsAnalyzing] = useState(false);
  const [mounted, setMounted] = useState(false);

  useEffect(() => {
    setMounted(true);
  }, []);

  const onDrop = useCallback((acceptedFiles: File[]) => {
    const file = acceptedFiles[0];
    if (file) {
      const reader = new FileReader();
      reader.onload = () => {
        const imageData = reader.result as string;
        setImage(imageData);
        analyzeTradingChart(imageData);
      };
      reader.readAsDataURL(file);
    }
  }, []);

  const { getRootProps, getInputProps, isDragActive } = useDropzone({
    onDrop,
    accept: {
      'image/*': ['.png', '.jpg', '.jpeg']
    },
    maxFiles: 1
  });

  const analyzeTradingChart = async (imageData: string) => {
    setIsAnalyzing(true);
    try {
      const tensor = await preprocessImage(imageData);
      const patterns = await detectChartPatterns(tensor);
      setPrediction(patterns);
      tensor.dispose(); // Clean up tensor memory
    } catch (error) {
      console.error('Error analyzing chart:', error);
      setPrediction(null);
    } finally {
      setIsAnalyzing(false);
    }
  };

  // Return null during SSR
  if (!mounted) {
    return null;
  }

  return (
    <main className="min-h-screen bg-gray-50 p-8">
      <div className="max-w-4xl mx-auto">
        <h1 className="text-4xl font-bold text-center mb-8">AI Trading Chart Analyzer</h1>
        
        <div className="bg-white rounded-lg shadow-lg p-8 mb-8">
          <div 
            {...getRootProps()} 
            className={`border-2 border-dashed rounded-lg p-8 text-center cursor-pointer
              ${isDragActive ? 'border-blue-500 bg-blue-50' : 'border-gray-300'}`}
          >
            <input {...getInputProps()} />
            <p className="text-lg text-gray-600">
              {isDragActive
                ? "Drop the trading chart here"
                : "Drag & drop a trading chart, or click to select"}
            </p>
          </div>
        </div>

        {isAnalyzing && (
          <div className="text-center p-4">
            <div className="animate-spin rounded-full h-12 w-12 border-b-2 border-blue-500 mx-auto"></div>
            <p className="mt-4 text-gray-600">Analyzing chart patterns...</p>
          </div>
        )}

        {image && !isAnalyzing && (
          <div className="bg-white rounded-lg shadow-lg p-8">
            <div className="grid grid-cols-1 md:grid-cols-2 gap-8">
              <div>
                <h2 className="text-2xl font-semibold mb-4">Uploaded Chart</h2>
                <div className="relative h-64 w-full">
                  <Image
                    src={image}
                    alt="Trading Chart"
                    fill
                    className="object-contain"
                  />
                </div>
              </div>
              <div>
                <h2 className="text-2xl font-semibold mb-4">Analysis Results</h2>
                <ChartAnalysis prediction={prediction} />
              </div>
            </div>
          </div>
        )}
      </div>
    </main>
  );
}