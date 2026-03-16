import "server-only";
import yahooFinance from "yahoo-finance2";
import * as tf from "@tensorflow/tfjs";
import { RSI, MACD, ATR, BollingerBands, EMA } from "technicalindicators";
import { subDays, startOfDay, isSameDay, getHours } from "date-fns";

// --- Types ---
export interface PredictionResult {
  symbol: string;
  lastKnownPrice: number;
  predictedEodClose: number;
  metrics: {
    mae: number;
    r2: number;
  };
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

export async function predictSymbol(symbol: string): Promise<PredictionResult> {
  // Set seed equivalent (TFJS requires global handling, usually handled at app start)
  // tf.random.setSeed(42);

  console.log(`Fetching data for ${symbol}...`);

  // 1. Data Collection
  const endDate = new Date();
  const startDate = subDays(endDate, 60);

  // Yahoo Finance fetch
  const rawData = await yahooFinance.historical(symbol, {
    period1: startDate,
    interval: "1h", // 1 hour interval
  });

  if (!rawData || rawData.length === 0) {
    throw new Error(`No data found for symbol '${symbol}'`);
  }

  // Resample to 4H
  let df: Candle[] = resampleTo4H(rawData as YahooFinanceData[]);

  // 2. Feature Engineering
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

  // 3. Target Definition & Data Splitting
  const lastDate = df[df.length - 1].date;
  const predictionDayDate = startOfDay(lastDate);

  const historicalData = df.filter((c) => c.date < predictionDayDate);
  const predictionData = df.filter((c) => isSameDay(c.date, predictionDayDate));

  if (predictionData.length === 0) {
    throw new Error(
      "No data available for the current day to make a prediction."
    );
  }

  // Create daily close map for historical data
  // In Python: historical_df.resample('D')['close'].last()
  const dailyCloseMap = new Map<number, number>(); // timestamp -> close

  // Helper to group by day and find last close
  const tempGroups = new Map<number, number>(); // day timestamp -> last close seen
  historicalData.forEach((c) => {
    const dayTs = startOfDay(c.date).getTime();
    tempGroups.set(dayTs, c.close); // Since we iterate in order, last set is last close
  });

  // Assign target
  // historical_df['target_eod_close']
  const historicalWithTarget = historicalData
    .filter((c) => {
      const dayTs = startOfDay(c.date).getTime();
      return tempGroups.has(dayTs);
    })
    .map((c) => ({
      ...c,
      target_eod_close: tempGroups.get(startOfDay(c.date).getTime())!,
    }));

  if (historicalWithTarget.length === 0) {
    throw new Error(
      "Not enough historical data to create a training set after filtering. Try a longer date range."
    );
  }

  // 4. Data Preparation for RNN
  const featuresToExclude = [
    "open",
    "high",
    "low",
    "close",
    "volume",
    "target_eod_close",
    "date",
  ];

  // Extract feature names dynamically
  const featureNames = Object.keys(historicalWithTarget[0]).filter(
    (k) => !featuresToExclude.includes(k)
  );

  const X_raw = historicalWithTarget.map((row) =>
    featureNames.map((name) => row[name])
  );
  const y_raw = historicalWithTarget.map((row) => [row.target_eod_close]);

  // Split train/test (80/20)
  const trainSize = Math.floor(X_raw.length * 0.8);
  const X_train_raw = X_raw.slice(0, trainSize);
  const X_test_raw = X_raw.slice(trainSize);
  const y_train_raw = y_raw.slice(0, trainSize);
  const y_test_raw = y_raw.slice(trainSize);

  // Scale
  const xScaler = new StandardScaler();
  xScaler.fit(X_train_raw);
  const X_train_scaled = xScaler.transform(X_train_raw);
  const X_test_scaled = xScaler.transform(X_test_raw);

  const yScaler = new MinMaxScaler();
  yScaler.fit(y_train_raw);
  const y_train_scaled = yScaler.transform(y_train_raw);
  const y_test_scaled = yScaler.transform(y_test_raw);

  // Sequences
  const sequenceLength = 10;

  const [X_train, y_train] = createSequences(
    X_train_scaled,
    y_train_scaled,
    sequenceLength
  );
  const [X_test, y_test] = createSequences(
    X_test_scaled,
    y_test_scaled,
    sequenceLength
  );

  console.log(`X_train shape: [${X_train.shape}]`);

  // 5. Model Training
  const model = tf.sequential();

  // SimpleRNN layer 1
  model.add(
    tf.layers.simpleRNN({
      units: 50,
      activation: "relu",
      returnSequences: true,
      inputShape: [sequenceLength, featureNames.length],
    })
  );
  model.add(tf.layers.dropout({ rate: 0.2 }));

  // SimpleRNN layer 2
  model.add(
    tf.layers.simpleRNN({
      units: 50,
      activation: "relu",
      returnSequences: false,
    })
  );
  model.add(tf.layers.dropout({ rate: 0.2 }));

  // Dense layers
  model.add(tf.layers.dense({ units: 25, activation: "relu" }));
  model.add(tf.layers.dense({ units: 1, activation: "linear" }));

  model.compile({ optimizer: "adam", loss: "meanSquaredError" });

  await model.fit(X_train, y_train, {
    epochs: 50,
    batchSize: 32,
    validationData: [X_test, y_test],
    verbose: 0, // Set to 1 to see logs in server console
  });

  // 6. Evaluation
  const yPredTensor = model.predict(X_test) as tf.Tensor;
  const yPredScaled = (await yPredTensor.array()) as number[][];
  const yPred = yScaler.inverseTransform(yPredScaled);
  const yTestInv = yScaler.inverseTransform(
    y_test_scaled.slice(sequenceLength)
  ); // Align with sequence slicing

  // Calculate Metrics (Manual implementation for simple MAE/R2)
  const mae = calculateMAE(yTestInv, yPred);
  const r2 = calculateR2(yTestInv, yPred);

  console.log(`MAE: ${mae.toFixed(4)}, R2: ${r2.toFixed(4)}`);

  // 7. Predict Today
  // Need last sequenceLength from the FULL dataset (df)
  // We need to re-extract features from 'df' (which contains prediction_df data too)
  const fullFeaturesRaw = df.map((row) =>
    featureNames.map((name) => row[name])
  );
  const latestSequenceRaw = fullFeaturesRaw.slice(-sequenceLength);

  // Scale using existing X scaler
  const latestSequenceScaled = xScaler.transform(latestSequenceRaw);

  // Create tensor (1, seq_len, features)
  const inputForPrediction = tf.tensor3d([latestSequenceScaled]);

  const todaysPredScaledTensor = model.predict(inputForPrediction) as tf.Tensor;
  const todaysPredScaled = (await todaysPredScaledTensor.array()) as number[][];
  const todaysPred = yScaler.inverseTransform(todaysPredScaled);

  // Cleanup tensors
  X_train.dispose();
  y_train.dispose();
  X_test.dispose();
  y_test.dispose();
  inputForPrediction.dispose();
  todaysPredScaledTensor.dispose();
  yPredTensor.dispose();

  return {
    symbol: symbol,
    lastKnownPrice: df[df.length - 1].close,
    predictedEodClose: todaysPred[0][0],
    metrics: { mae, r2 },
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

function calculateMAE(yTrue: number[][], yPred: number[][]): number {
  let sum = 0;
  for (let i = 0; i < yTrue.length; i++) {
    sum += Math.abs(yTrue[i][0] - yPred[i][0]);
  }
  return sum / yTrue.length;
}

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
