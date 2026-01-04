import axios from 'axios';
import type { MatchRequest, MatchResponse } from './types'; 

// Kosongkan base URL jika dideploy ke satu domain Vercel yang sama
// Karena vercel.json akan menangani redirect /api/ ke backend
const API_BASE_URL = ''; 

export const postMatchRequest = async (formData: FormData): Promise<MatchResponse> => {
  const response = await axios.post<MatchResponse>(`${API_BASE_URL}/api/match`, formData, {
    headers: {
      'Content-Type': 'multipart/form-data',
    },
  });
  return response.data;
};