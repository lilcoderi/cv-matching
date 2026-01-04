// frontend/src/type.ts
export interface CVResult {
  filename: string;
  score: number;
  percentage: number;
  status: "Cocok" | "Tidak Cocok";
}

export interface MatchResponse {
  results: CVResult[];
}