#!/usr/bin/env python3
import subprocess
import time
import os

class MediaPlayer:
    def __init__(self):
        self.process = None

    def play_music(self, genre="relaxing"):
        """
        Plays music using ffplay and internet radio streams.
        """
        self.stop_music() # Stop any existing playback
        
        print(f"\n🎵 Starting {genre} music player (ffplay)...")
        
        # Direct stream URLs (Internet Radio)
        streams = {
            "relaxing": "http://stream.zeno.fm/0r0xa854rp8uv", # Classical/Relaxing
            "upbeat": "http://stream.zeno.fm/f3wvbbqmdg8uv",   # Pop/Upbeat
            "jazz": "http://ice1.somafm.com/sonicuniverse-128-mp3", # Jazz
            "nature": "http://ice1.somafm.com/dronezone-128-mp3",   # Ambient/Nature
            "default": "http://stream.zeno.fm/0r0xa854rp8uv"
        }
        
        stream_url = streams.get(genre, streams["default"])
        
        try:
            # -nodisp: No display window
            # -autoexit: Exit when stream ends (though radio doesn't end)
            # -loglevel quiet: Suppress output
            cmd = ["ffplay", "-nodisp", "-autoexit", "-loglevel", "quiet", stream_url]
            
            # Start in background
            self.process = subprocess.Popen(cmd)
            return True
        except Exception as e:
            print(f"Error playing music: {e}")
            return False

    def stop_music(self):
        if self.process:
            print("Stopping music...")
            self.process.terminate()
            self.process = None
