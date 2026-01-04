import axios from 'axios';
import type { MatchResponse } from './types'; 

export const postMatchRequest = async (formData: FormData): Promise<MatchResponse> => {
  const response = await axios.post<MatchResponse>(`/api/match`, formData, {
    headers: {
      'Content-Type': 'multipart/form-data',
    },
  });
  return response.data;
};