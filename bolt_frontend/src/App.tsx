import { useState } from 'react';
import { useToast } from '@/hooks/use-toast';
import { SearchForm } from '@/components/SearchForm';
import { SongList } from '@/components/SongList';
import { getSongRecommendations } from '@/services/api';
import type { Song } from '@/types/song';

function App() {
  const [songs, setSongs] = useState<Song[]>([]);
  const [isLoading, setIsLoading] = useState(false);
  const { toast } = useToast();

  const handleSubmit = async (query: string) => {
    setIsLoading(true);
    try {
      const response = await getSongRecommendations(query);
      setSongs(response.recommendations);
    } catch (error) {
      toast({
        title: 'Error',
        description: 'Failed to fetch song recommendations. Please try again.',
        variant: 'destructive',
      });
    } finally {
      setIsLoading(false);
    }
  };

  return (
    <main className="min-h-screen bg-background p-4 md:p-8">
      <div className="container mx-auto space-y-6">
        <SearchForm onSubmit={handleSubmit} isLoading={isLoading} />
        <SongList songs={songs} />
      </div>
    </main>
  );
}

export default App;