import "server-only";
import YahooFinance from "yahoo-finance2";
import * as tf from "@tensorflow/tfjs-node";
import { RSI, MACD, ATR, BollingerBands, EMA } from "technicalindicators";
import { subDays, startOfDay, isSameDay, getHours } from "date-fns";
import path from "path";
import fs from "fs";

// --- Types ---
export interface PredictionResult {
  symbol: string;
  lastKnownPrice: number;
  predictedEodClose: number;
  // Metrics are calculated during offline training, not on each prediction.
  // You can include them here if you load them from a file.
  metrics?: { mae: number; r2: number };
}

interface Candle {
  date: Date;
  open: number;
  high: number;
  low: number;
  close: number;
  volume: number;
  [key: string]: any; // Allow for dynamic feature properties
}

interface YahooFinanceData {
  date: Date;
  open: number;
  high: number;
  low: number;
  close: number;
  volume: number;
}

// --- Scaler Utility Classes ---
class StandardScaler {
  private mean: number[] = [];
  private std: number[] = [];

  fit(data: number[][]) {
    const n = data.length;
    if (n === 0) return;
    const features = data[0].length;

    // Calculate Mean
    this.mean = new Array(features).fill(0);
    for (const row of data) {
      for (let i = 0; i < features; i++) this.mean[i] += row[i];
    }
    this.mean = this.mean.map((sum) => sum / n);

    // Calculate Std Dev
    this.std = new Array(features).fill(0);
    for (const row of data) {
      for (let i = 0; i < features; i++) {
        this.std[i] += Math.pow(row[i] - this.mean[i], 2);
      }
    }
    this.std = this.std.map((sum) => Math.sqrt(sum / n) || 1); // Avoid div by zero
  }

  toJSON() {
    return { mean: this.mean, std: this.std };
  }

  static fromJSON(json: { mean: number[]; std: number[] }): StandardScaler {
    const scaler = new StandardScaler();
    if (!json.mean || !json.std) throw new Error("Invalid StandardScaler JSON");
    scaler.mean = json.mean;
    scaler.std = json.std;
    return scaler;
  }

  transform(data: number[][]): number[][] {
    return data.map((row) =>
      row.map((val, i) => (val - this.mean[i]) / this.std[i])
    );
  }
}

class MinMaxScaler {
  private min: number[] = [];
  private max: number[] = [];

  fit(data: number[][]) {
    if (data.length === 0) return;
    const features = data[0].length;
    this.min = new Array(features).fill(Infinity);
    this.max = new Array(features).fill(-Infinity);

    for (const row of data) {
      for (let i = 0; i < features; i++) {
        if (row[i] < this.min[i]) this.min[i] = row[i];
        if (row[i] > this.max[i]) this.max[i] = row[i];
      }
    }
  }

  toJSON() {
    return { min: this.min, max: this.max };
  }

  static fromJSON(json: { min: number[]; max: number[] }): MinMaxScaler {
    const scaler = new MinMaxScaler();
    if (!json.min || !json.max) throw new Error("Invalid MinMaxScaler JSON");
    scaler.min = json.min;
    scaler.max = json.max;
    return scaler;
  }

  transform(data: number[][]): number[][] {
    return data.map((row) =>
      row.map(
        (val, i) => (val - this.min[i]) / (this.max[i] - this.min[i] || 1)
      )
    );
  }

  inverseTransform(data: number[][]): number[][] {
    return data.map((row) =>
      row.map((val, i) => val * (this.max[i] - this.min[i]) + this.min[i])
    );
  }
}

// --- Main Logic ---
const MODEL_DIR = path.join(process.cwd(), "public", "model");
const MODEL_PATH = `file://${path.join(MODEL_DIR, "model.json")}`;
const SCALERS_PATH = path.join(MODEL_DIR, "scalers.json");

// Cache for the loaded model and scalers
let loadedArtifacts: {
  model: tf.LayersModel;
  xScaler: StandardScaler;
  yScaler: MinMaxScaler;
  featureNames: string[];
} | null = null;

async function loadModelAndScalers() {
  if (loadedArtifacts) return loadedArtifacts;

  if (!fs.existsSync(path.join(MODEL_DIR, "model.json"))) {
    throw new Error(
      `Model not found at ${MODEL_DIR}. Please run the training script to generate model files.`
    );
  }

  console.log("Loading model and scalers...");
  const model = await tf.loadLayersModel(MODEL_PATH);
  const scalersData = JSON.parse(fs.readFileSync(SCALERS_PATH, "utf-8"));

  const xScaler = StandardScaler.fromJSON(scalersData.xScaler);
  const yScaler = MinMaxScaler.fromJSON(scalersData.yScaler);
  const featureNames = scalersData.featureNames as string[];

  loadedArtifacts = { model, xScaler, yScaler, featureNames };
  console.log("Model and scalers loaded successfully.");
  return loadedArtifacts;
}

const yahooFinance = new YahooFinance();

export async function predictSymbol(symbol: string): Promise<PredictionResult> {
  console.log(`Fetching data for ${symbol}...`);

  // 1. Data Collection
  const endDate = new Date();
  // Fetch enough data for feature calculation (EMA 50 is longest) + sequence length
  const startDate = subDays(endDate, 30); // 30 days should be sufficient
  const result = await yahooFinance.chart(symbol, {
    period1: startDate,
    interval: "1h", // 1 hour interval
  });
  const rawData = result.quotes;

  if (!rawData || rawData.length === 0) {
    throw new Error(`No data found for symbol '${symbol}'`);
  }

  // 2. Feature Engineering
  let df: Candle[] = resampleTo4H(rawData as YahooFinanceData[]);

  // --- Start of feature engineering ---
  // Need arrays for technicalindicators lib
  const close = df.map((c) => c.close);
  const high = df.map((c) => c.high);
  const low = df.map((c) => c.low);

  // RSI (14)
  const rsi = RSI.calculate({ values: close, period: 14 });
  // MACD (12, 26)
  const macd = MACD.calculate({
    values: close,
    fastPeriod: 12,
    slowPeriod: 26,
    signalPeriod: 9,
    SimpleMAOscillator: false,
    SimpleMASignal: false,
  });
  // ATR (14)
  const atr = ATR.calculate({ high, low, close, period: 14 });
  // BBands (20)
  const bbands = BollingerBands.calculate({
    values: close,
    period: 20,
    stdDev: 2,
  });
  // EMA (50)
  const ema50 = EMA.calculate({ values: close, period: 50 });

  // Integrate indicators back into DataFrame.
  // Note: Indicators result in shorter arrays. We align from the end.
  // We drop rows where any indicator is missing (similar to df.dropna()).

  const minLen = Math.min(
    df.length,
    rsi.length,
    macd.length,
    atr.length,
    bbands.length,
    ema50.length
  );

  // Trim df to match simplest common length (from the end)
  df = df.slice(df.length - minLen);
  const rsiSliced = rsi.slice(rsi.length - minLen);
  const macdSliced = macd.slice(macd.length - minLen);
  const atrSliced = atr.slice(atr.length - minLen);
  const bbandsSliced = bbands.slice(bbands.length - minLen);
  const emaSliced = ema50.slice(ema50.length - minLen);

  // Assign features
  df.forEach((row, i) => {
    row.RSI_14 = rsiSliced[i];
    row.MACD_12_26 = macdSliced[i].MACD;
    row.MACD_signal = macdSliced[i].signal; // Py script usually just uses append=True which adds all columns
    row.ATR_14 = atrSliced[i];
    row.BBL_20 = bbandsSliced[i].lower;
    row.BBM_20 = bbandsSliced[i].middle;
    row.BBU_20 = bbandsSliced[i].upper;
    row.EMA_50 = emaSliced[i];
  });

  // Lagged Features (1 to 5)
  // This creates NaN for the first 5 rows, so we slice again
  const lags = 5;
  for (let i = lags; i < df.length; i++) {
    for (let lag = 1; lag <= lags; lag++) {
      df[i][`price_change_lag_${lag}`] = df[i].close - df[i - lag].close;
    }
  }
  df = df.slice(lags); // Drop rows with undefined lags
  // --- End of feature engineering ---

  if (df.length < 10) {
    // Sequence length is 10
    throw new Error(
      "Not enough data to make a prediction after feature engineering."
    );
  }

  // 3. Load Model and Scalers
  const { model, xScaler, yScaler, featureNames } = await loadModelAndScalers();

  // 4. Prepare data for prediction
  const sequenceLength = 10;
  const fullFeaturesRaw = df.map((row) =>
    featureNames.map((name) => row[name])
  );
  const latestSequenceRaw = fullFeaturesRaw.slice(-sequenceLength);

  // Scale using loaded X scaler
  const latestSequenceScaled = xScaler.transform(latestSequenceRaw);

  // Create tensor (1, seq_len, features)
  const inputForPrediction = tf.tensor3d([latestSequenceScaled]);

  // 5. Make Prediction
  const todaysPredScaledTensor = model.predict(inputForPrediction) as tf.Tensor;
  const todaysPredScaled = (await todaysPredScaledTensor.array()) as number[][];
  const todaysPred = yScaler.inverseTransform(todaysPredScaled);

  // Cleanup tensors
  inputForPrediction.dispose();
  todaysPredScaledTensor.dispose();

  return {
    symbol: symbol,
    lastKnownPrice: df[df.length - 1].close,
    predictedEodClose: todaysPred[0][0],
  };
}

// --- Helpers ---

function resampleTo4H(data: YahooFinanceData[]): Candle[] {
  // Assumes data is sorted by date
  const resampled: Candle[] = [];
  let currentBucket: YahooFinanceData[] = [];
  let bucketStartTime: number | null = null;

  for (const row of data) {
    const date = new Date(row.date);
    // Calculate 4H bucket floor (0, 4, 8, 12, 16, 20)
    const hour = getHours(date);
    const bucketHour = Math.floor(hour / 4) * 4;
    const bucketDate = startOfDay(date);
    bucketDate.setHours(bucketHour);
    const bucketTime = bucketDate.getTime();

    if (bucketStartTime === null) bucketStartTime = bucketTime;

    if (bucketTime !== bucketStartTime) {
      // Close previous bucket
      if (currentBucket.length > 0) {
        resampled.push(
          aggregateBucket(currentBucket, new Date(bucketStartTime))
        );
      }
      currentBucket = [];
      bucketStartTime = bucketTime;
    }
    currentBucket.push(row);
  }
  // Push last bucket
  if (currentBucket.length > 0 && bucketStartTime !== null) {
    resampled.push(aggregateBucket(currentBucket, new Date(bucketStartTime)));
  }
  return resampled;
}

// This helper function remains unchanged
function aggregateBucket(rows: YahooFinanceData[], date: Date): Candle {
  return {
    date: date,
    open: rows[0].open,
    high: Math.max(...rows.map((r) => r.high)),
    low: Math.min(...rows.map((r) => r.low)),
    close: rows[rows.length - 1].close,
    volume: rows.reduce((sum, r) => sum + r.volume, 0),
  };
}

// This helper function is now only needed for the training script
function createSequences(
  X_data: number[][],
  y_data: number[][],
  seq_length: number
): [tf.Tensor3D, tf.Tensor2D] {
  const X_seq: number[][][] = [];
  const y_seq: number[][] = [];
  for (let i = 0; i < X_data.length - seq_length; i++) {
    X_seq.push(X_data.slice(i, i + seq_length));
    y_seq.push(y_data[i + seq_length]);
  }
  return [tf.tensor3d(X_seq), tf.tensor2d(y_seq)];
}

// This helper function is now only needed for the training script
function calculateMAE(yTrue: number[][], yPred: number[][]): number {
  let sum = 0;
  for (let i = 0; i < yTrue.length; i++) {
    sum += Math.abs(yTrue[i][0] - yPred[i][0]);
  }
  return sum / yTrue.length;
}

// This helper function is now only needed for the training script
function calculateR2(yTrue: number[][], yPred: number[][]): number {
  const meanTrue = yTrue.reduce((sum, val) => sum + val[0], 0) / yTrue.length;
  let ssTot = 0;
  let ssRes = 0;
  for (let i = 0; i < yTrue.length; i++) {
    ssTot += Math.pow(yTrue[i][0] - meanTrue, 2);
    ssRes += Math.pow(yTrue[i][0] - yPred[i][0], 2);
  }
  return 1 - ssRes / ssTot;
}
