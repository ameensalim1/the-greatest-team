import type { Song } from '@/types/song';
import { SongItem } from '@/components/SongItem';
import { ScrollArea } from '@/components/ui/scroll-area';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';

interface SongListProps {
  songs: Song[];
}

export function SongList({ songs }: SongListProps) {
  if (!songs.length) return null;

  return (
    <Card className="w-full max-w-2xl">
      <CardHeader>
        <CardTitle>Recommended Songs</CardTitle>
      </CardHeader>
      <CardContent>
        <ScrollArea className="h-[400px] pr-4">
          <div className="space-y-2">
            {songs.map((song, index) => (
              <SongItem key={`${song.artist_name}-${song.track_name}-${index}`} song={song} />
            ))}
          </div>
        </ScrollArea>
      </CardContent>
    </Card>
  );
}