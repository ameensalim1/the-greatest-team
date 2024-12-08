import type { Song } from '@/types/song';
import { Music2 } from 'lucide-react';

interface SongItemProps {
  song: Song;
}

export function SongItem({ song }: SongItemProps) {
  return (
    <div className="flex items-center gap-3 p-4 rounded-lg bg-card hover:bg-accent/50 transition-colors">
      <Music2 className="h-5 w-5 text-primary/70" />
      <div>
        <h3 className="font-medium text-lg">{song.track_name}</h3>
        <p className="text-sm text-muted-foreground">{song.artist_name}</p>
      </div>
    </div>
  );
}