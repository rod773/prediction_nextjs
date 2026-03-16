import { NextResponse } from "next/server";
import { predictSymbol } from "@/lib/predictRNN";

export async function POST(request: Request) {
  try {
    const { symbol } = await request.json();

    if (!symbol || typeof symbol !== "string") {
      return NextResponse.json(
        { error: "Symbol is required and must be a string." },
        { status: 400 }
      );
    }

    console.log(`[API] Received prediction request for: ${symbol}`);
    const predictionResult = await predictSymbol(symbol.toUpperCase());

    return NextResponse.json(predictionResult);
  } catch (error) {
    console.error("[API_ERROR]", error);
    const errorMessage =
      error instanceof Error
        ? error.message
        : "An unknown error occurred during prediction.";

    return NextResponse.json({ error: errorMessage }, { status: 500 });
  }
}
