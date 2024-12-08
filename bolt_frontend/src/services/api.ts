import axios from 'axios';
import type { SongRecommendationResponse } from '@/types/song';

const API_BASE_URL = 'http://localhost:5000';

export const getSongRecommendations = async (query: string): Promise<SongRecommendationResponse> => {
  const response = await axios.post<SongRecommendationResponse>(
    `${API_BASE_URL}/recommend-by-genre`,
    { query }
  );
  return response.data;
};