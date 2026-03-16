import { PredictionForm } from "@/components/PredictionForm";

export default function Home() {
  return (
    <main className="flex min-h-screen flex-col items-center justify-center p-4 sm:p-8 md:p-24 bg-background">
      <div className="absolute top-4 right-4 text-sm text-muted-foreground">
        Gemini Code Assist
      </div>
      <PredictionForm />
    </main>
  );
}
