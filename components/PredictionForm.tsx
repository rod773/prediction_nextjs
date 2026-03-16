"use client";

import { useState } from "react";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import {
  Card,
  CardContent,
  CardDescription,
  CardFooter,
  CardHeader,
  CardTitle,
} from "@/components/ui/card";
import { useToast } from "@/components/ui/use-toast";
import { PredictionResult } from "@/lib/predictRNN"; // Import the type

export function PredictionForm() {
  const [symbol, setSymbol] = useState("AAPL");
  const [isLoading, setIsLoading] = useState(false);
  const [result, setResult] = useState<PredictionResult | null>(null);
  const { toast } = useToast();

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    setIsLoading(true);
    setResult(null);

    try {
      const response = await fetch("/api/predict", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ symbol }),
      });

      const data = await response.json();

      if (!response.ok) {
        throw new Error(data.error || "Something went wrong");
      }

      setResult(data);
    } catch (error) {
      const errorMessage =
        error instanceof Error ? error.message : "An unknown error occurred";
      toast({
        variant: "destructive",
        title: "Prediction Failed",
        description: errorMessage,
      });
    } finally {
      setIsLoading(false);
    }
  };

  return (
    <div className="w-full max-w-md space-y-6">
      <Card>
        <CardHeader>
          <CardTitle>Stock Price Predictor</CardTitle>
          <CardDescription>
            Enter a stock symbol to predict its end-of-day closing price using
            an RNN model.
          </CardDescription>
        </CardHeader>
        <form onSubmit={handleSubmit}>
          <CardContent>
            <div className="space-y-2">
              <Label htmlFor="symbol">Stock Symbol</Label>
              <Input
                id="symbol"
                placeholder="e.g., AAPL, GOOG, BTC-USD"
                value={symbol}
                onChange={(e) => setSymbol(e.target.value)}
                disabled={isLoading}
                required
              />
            </div>
          </CardContent>
          <CardFooter>
            <Button type="submit" className="w-full" disabled={isLoading}>
              {isLoading ? "Predicting..." : "Predict Today's Close"}
            </Button>
          </CardFooter>
        </form>
      </Card>

      {result && (
        <Card>
          <CardHeader>
            <CardTitle>Prediction for {result.symbol}</CardTitle>
            <CardDescription>
              Based on the last 60 days of hourly data.
            </CardDescription>
          </CardHeader>
          <CardContent className="grid gap-4">
            <div className="flex items-center justify-between">
              <span className="text-muted-foreground">Last Known Price</span>
              <strong>${result.lastKnownPrice.toFixed(2)}</strong>
            </div>
            <div className="flex items-center justify-between text-lg">
              <span className="text-muted-foreground">Predicted EOD Close</span>
              <strong className="text-green-500">
                ${result.predictedEodClose.toFixed(2)}
              </strong>
            </div>
            <hr />
            <div className="text-sm text-muted-foreground space-y-2">
              <p className="font-semibold">Model Performance (on test data):</p>
              <div className="flex items-center justify-between">
                <span>Mean Absolute Error (MAE):</span>
                <span>${result.metrics.mae.toFixed(4)}</span>
              </div>
              <div className="flex items-center justify-between">
                <span>R-squared (R²):</span>
                <span>{result.metrics.r2.toFixed(4)}</span>
              </div>
            </div>
          </CardContent>
        </Card>
      )}
    </div>
  );
}
