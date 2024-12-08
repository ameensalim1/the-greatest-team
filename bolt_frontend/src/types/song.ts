export interface Song {
  artist_name: string;
  track_name: string;
}

export interface SongRecommendationResponse {
  recommendations: Song[];
}