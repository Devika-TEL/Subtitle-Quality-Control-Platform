import os
import re
import subprocess
import numpy as np
import librosa
from scipy import signal
from scipy.ndimage import binary_erosion, binary_dilation
from difflib import SequenceMatcher
import whisper

class SubtitleSynchronizer:
    def __init__(self, video_path, subtitle_path, output_path=None):
        self.video_path = video_path
        self.subtitle_path = subtitle_path
        self.output_path = output_path or self._generate_output_path()
        self.audio_sample_rate = 22050
        self.hop_length = 512
        self.frame_length = 2048
        
    def _generate_output_path(self):
        """Generate output path with _synced suffix"""
        base, ext = os.path.splitext(self.subtitle_path)
        return f"{base}_synced{ext}"
    
    # STEP 1: Audio Extraction
    #---------------------------
    def extract_audio(self, temp_audio_path="temp_audio.wav"):
        """Extract audio from video file using ffmpeg"""
        try:
            cmd = [
                'ffmpeg', '-i', self.video_path, 
                '-vn', '-acodec', 'pcm_s16le', 
                '-ar', str(self.audio_sample_rate), 
                '-ac', '1', '-y', temp_audio_path
            ]
            subprocess.run(cmd, check=True, capture_output=True)
            return temp_audio_path
        except subprocess.CalledProcessError as e:
            raise Exception(f"Error extracting audio: {e}")
    
    # STEP 2: Subtitle Parsing
    #----------------------------
    def parse_subtitle_file(self):
        """Parse subtitle file and detect format"""
        with open(self.subtitle_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        if content.strip().startswith('WEBVTT'):
            return self._parse_vtt(content)
        elif re.search(r'\d+\n\d{2}:\d{2}:\d{2},\d{3}', content):
            return self._parse_srt(content)
        else:
            raise ValueError("Unsupported subtitle format")
    
    def _time_to_seconds(self, hours, minutes, seconds, milliseconds):
        """Convert time components to total seconds"""
        return int(hours) * 3600 + int(minutes) * 60 + int(seconds) + int(milliseconds) / 1000

    def _parse_srt(self, content):
        """Parse SRT subtitle format"""
        subtitles = []
        blocks = re.split(r'\n\s*\n', content.strip())
        
        for block in blocks:
            lines = block.strip().split('\n')
            if len(lines) >= 3:
                try:
                    index = int(lines[0])
                    time_line = lines[1]
                    text = '\n'.join(lines[2:])
                    
                    # Parse time format: 00:00:20,000 --> 00:00:24,400
                    time_match = re.match(r'(\d{2}):(\d{2}):(\d{2}),(\d{3})\s*-->\s*(\d{2}):(\d{2}):(\d{2}),(\d{3})', time_line)
                    if time_match:
                        start_time = self._time_to_seconds(*time_match.groups()[:4])
                        end_time = self._time_to_seconds(*time_match.groups()[4:])
                        
                        subtitles.append({
                            'index': index,
                            'start': start_time,
                            'end': end_time,
                            'text': text,
                            'format': 'srt'
                        })
                except (ValueError, AttributeError):
                    continue
        
        return subtitles
    
    def _parse_vtt(self, content):
        """Parse WebVTT subtitle format"""
        subtitles = []
        lines = content.split('\n')
        i = 0
        index = 1
        
        while i < len(lines):
            line = lines[i].strip()
            
            # Look for timestamp line
            time_match = re.match(r'(\d{2}):(\d{2}):(\d{2})\.(\d{3})\s*-->\s*(\d{2}):(\d{2}):(\d{2})\.(\d{3})', line)
            if time_match:
                start_time = self._time_to_seconds(*time_match.groups()[:4])
                end_time = self._time_to_seconds(*time_match.groups()[4:])
                
                # Collect subtitle text
                text_lines = []
                i += 1
                while i < len(lines) and lines[i].strip():
                    text_lines.append(lines[i].strip())
                    i += 1
                
                if text_lines:
                    subtitles.append({
                        'index': index,
                        'start': start_time,
                        'end': end_time,
                        'text': '\n'.join(text_lines),
                        'format': 'vtt'
                    })
                    index += 1
            
            i += 1
        
        return subtitles
    
    # STEP 3: Speech Detection
    #----------------------------
    def detect_speech_segments(self, audio_path):
        """Detect speech segments using energy and spectral features"""
        # Load audio
        audio, sr = librosa.load(audio_path, sr=self.audio_sample_rate)
        
        # Method 1: Energy-based detection
        frame_length = int(0.025 * sr)  # 25ms frames
        hop_length = int(0.010 * sr)    # 10ms hop

        # ------Audio Feature Extraction-----
        # Calculate short-time energy
        energy = librosa.feature.rms(y=audio, frame_length=frame_length, hop_length=hop_length)[0]
        
        # Calculate spectral centroid (indicates presence of speech)
        spectral_centroid = librosa.feature.spectral_centroid(y=audio, sr=sr, hop_length=hop_length)[0]
        
        # Calculate zero crossing rate
        zcr = librosa.feature.zero_crossing_rate(audio, frame_length=frame_length, hop_length=hop_length)[0]
        
        # -----Feature Normalization-----
        energy_norm = (energy - np.mean(energy)) / (np.std(energy) + 1e-8)
        centroid_norm = (spectral_centroid - np.mean(spectral_centroid)) / (np.std(spectral_centroid) + 1e-8)
        zcr_norm = (zcr - np.mean(zcr)) / (np.std(zcr) + 1e-8)
        
        # -----Speech Score Calculation-----
        # Speech typically has moderate energy, higher spectral centroid, and moderate ZCR
        speech_score = energy_norm + 0.3 * centroid_norm - 0.2 * zcr_norm
        
        # -----Adaptive Thresholding------
        threshold = np.percentile(speech_score, 60)  # Top 40% likely to be speech
        
        # Create binary speech mask
        speech_mask = speech_score > threshold
        
        # -----Morphological Operations-----
        # Remove small gaps and noise, to clean up the mask
        speech_mask = binary_dilation(speech_mask, iterations=2)
        speech_mask = binary_erosion(speech_mask, iterations=1)
        
        # Convert frame indices to time segments
        speech_segments = []
        frame_times = librosa.frames_to_time(np.arange(len(speech_mask)), sr=sr, hop_length=hop_length)
        
        in_speech = False
        start_time = 0
        
        for i, (is_speech, time) in enumerate(zip(speech_mask, frame_times)):
            if is_speech and not in_speech:
                start_time = time
                in_speech = True
            elif not is_speech and in_speech:
                if time - start_time > 0.1:  # Minimum 100ms segment
                    speech_segments.append((start_time, time))
                in_speech = False
        
        # Handle case where speech continues to end of audio
        if in_speech:
            speech_segments.append((start_time, frame_times[-1]))
        
        #-----Segment Merging-----
        merged_segments = self._merge_speech_segments(speech_segments, gap_threshold=0.3)
        
        return merged_segments
    
    def _merge_speech_segments(self, segments, gap_threshold=0.5):
        """Merge speech segments that are close together"""
        if not segments:
            return []
        
        merged = []
        current_start, current_end = segments[0]
        
        for start, end in segments[1:]:
            if start - current_end <= gap_threshold:
                current_end = end
            else:
                merged.append((current_start, current_end))
                current_start, current_end = start, end
        
        merged.append((current_start, current_end))
        return merged
    
    # STEP 4: Synchronization Calculation
    #-------------------------------------
    def calculate_sync_offset(self, subtitles, speech_segments, audio_path=None):
        """Calculate optimal sync offset using cross-correlation - SIMPLIFIED"""
        # Check for edge cases
        if not subtitles or not speech_segments:
            return 0
        
        # Use a flag to prevent infinite reconstruction loops
        is_recursive = getattr(self, '_is_recursive_call', False)
        
        # PATH 1: Zero Timestamp Detection
        if self._has_any_zero_timestamps(subtitles) and not is_recursive:
            print("Detected zero timestamps. Using Whisper-based reconstruction...")
            if audio_path:
                self._whisper_align_zero_timestamps(subtitles, audio_path)
                return "ZERO_TIMESTAMPS_RECONSTRUCTED"
            else:
                print("Warning: No audio path provided for Whisper alignment")
                return 0
        
        # Continue with normal cross-correlation sync for fully valid timestamps
        max_time = max(
            max([sub['end'] for sub in subtitles]) if subtitles else 0,
            max([seg[1] for seg in speech_segments]) if speech_segments else 0
        )
        
        if max_time == 0:
            return 0
        
        # PATH 2: Normal Cross-Correlation Sync
        
        time_resolution = 0.1  # 100ms resolution
        time_bins = int(max_time / time_resolution) + 1
        
        subtitle_signal = np.zeros(time_bins)
        speech_signal = np.zeros(time_bins)
        
        # Fill subtitle signal
        for sub in subtitles:
            start_bin = int(sub['start'] / time_resolution)
            end_bin = int(sub['end'] / time_resolution)
            subtitle_signal[start_bin:end_bin] = 1
        
        # Fill speech signal
        for start, end in speech_segments:
            start_bin = int(start / time_resolution)
            end_bin = int(end / time_resolution)
            speech_signal[start_bin:end_bin] = 1
        
        # Calculate cross-correlation
        correlation = signal.correlate(speech_signal, subtitle_signal, mode='full')
        
        # Find best alignment
        max_corr_idx = np.argmax(correlation)
        offset_bins = max_corr_idx - (len(subtitle_signal) - 1)
        offset_seconds = offset_bins * time_resolution
        
        return offset_seconds

    def _has_any_zero_timestamps(self, subtitles):
        """Check if ANY subtitle timestamps are zero"""
        return any(sub['start'] == 0 and sub['end'] == 0 for sub in subtitles)

    # STEP 5: Whisper-based Reconstruction
    #-------------------------------------
    def _whisper_align_zero_timestamps(self, subtitles, audio_path):
        """Use Whisper to align zero timestamps with actual speech"""
        print("Loading Whisper model...")
        try:
            model = whisper.load_model("base")  # Use base model for speed, can change to "small" or "medium"
        except Exception as e:
            print(f"Error loading Whisper model: {e}")
            print("Falling back to basic speech detection...")
            self._estimate_timing_from_zero_fallback(subtitles)
            return
        
        print("Transcribing audio with Whisper...")
        try:
            result = model.transcribe(audio_path, word_timestamps=True)
        except Exception as e:
            print(f"Error during Whisper transcription: {e}")
            print("Falling back to basic speech detection...")
            self._estimate_timing_from_zero_fallback(subtitles)
            return
        
        # Extract subtitle texts
        sub_texts = [sub['text'].strip().replace('\n', ' ') for sub in subtitles]
        
        # Flatten Whisper segments
        whisper_segments = self._flatten_whisper_segments(result['segments'])
        
        print(f"Found {len(whisper_segments)} Whisper segments for {len(sub_texts)} subtitles")
        
        # Align subtitles to Whisper segments
        aligned_subtitles = self._align_subs_to_whisper(sub_texts, whisper_segments)
        
        # Update original subtitles with aligned timings
        for i, (start, end, text) in enumerate(aligned_subtitles):
            if i < len(subtitles):
                subtitles[i]['start'] = start
                subtitles[i]['end'] = end
        
        # Print results
        print("Whisper-based alignment results:")
        successful_alignments = sum(1 for start, end, _ in aligned_subtitles if start != 0 or end != 0)
        print(f"Successfully aligned: {successful_alignments}/{len(subtitles)} subtitles")
        
        for i, (start, end, text) in enumerate(aligned_subtitles[:5]):
            if start != 0 or end != 0:
                print(f"  Subtitle {i+1}: {self._seconds_to_srt_time(start)} --> {self._seconds_to_srt_time(end)}")
            else:
                print(f"  Subtitle {i+1}: UNMATCHED - {text[:50]}...")
        
        if len(aligned_subtitles) > 5:
            print(f"  ... and {len(aligned_subtitles)-5} more")
    
    def _flatten_whisper_segments(self, segments):
        """Convert Whisper segments to simple (start, end, text) tuples"""
        return [(seg['start'], seg['end'], seg['text'].strip()) for seg in segments]

    def _align_subs_to_whisper(self, sub_texts, whisper_segments):
        """Align subtitle texts to Whisper segments using text similarity"""
        aligned_subs = []
        used = [False] * len(whisper_segments)
        
        for sub_text in sub_texts:
            best_match = None
            best_score = 0
            
            for i, (start, end, whisper_text) in enumerate(whisper_segments):
                if used[i]:
                    continue
                
                # Calculate text similarity
                score = SequenceMatcher(None, sub_text.lower(), whisper_text.lower()).ratio()
                
                if score > best_score:
                    best_score = score
                    best_match = (start, end, whisper_text, i)
            
            if best_match and best_score > 0.5:  # Minimum 50% similarity
                start, end, whisper_text, idx = best_match
                aligned_subs.append((start, end, sub_text))
                used[idx] = True
                print(f"Matched (score: {best_score:.2f}): '{sub_text[:30]}...' -> '{whisper_text[:30]}...'")
            else:
                # No good match found
                aligned_subs.append((0, 0, sub_text))
                print(f"No match found for: '{sub_text[:50]}...'")
        
        return aligned_subs

    def _estimate_timing_from_zero_fallback(self, subtitles):
        """Fallback method when Whisper fails"""
        print("Using fallback timing estimation...")
        current_time = 0
        
        for sub in subtitles:
            # Simple duration estimation
            word_count = len(sub['text'].split())
            duration = max(1.5, word_count / 2.5)  # Minimum 1.5s, ~2.5 words per second
            
            sub['start'] = current_time
            sub['end'] = current_time + duration
            current_time += duration + 0.5  # 0.5s gap between subtitles
        
        print("Applied fallback timing based on text length")
    
    # STEP 6: Main Synchronization Workflow
    #----------------------------------------
    def synchronize(self, is_recursive_call=False):
        """Main synchronization process - FIXED THRESHOLD"""
        if not is_recursive_call:
            print("Starting subtitle synchronization...")
        else:
            print("Running fine-tuning synchronization on reconstructed subtitles...")
        
        # Extract audio from video
        if not is_recursive_call:
            print("Extracting audio...")
            temp_audio = self.extract_audio()
        else:
            # Reuse existing audio file for recursive call
            temp_audio = "temp_audio.wav"
            if not os.path.exists(temp_audio):
                temp_audio = self.extract_audio()
        
        try:
            # Parse subtitle file
            print("Parsing subtitles...")
            subtitles = self.parse_subtitle_file()
            
            if not subtitles:
                raise ValueError("No valid subtitles found")
            
            # Detect speech segments
            if not is_recursive_call:
                print("Detecting speech segments...")
                speech_segments = self.detect_speech_segments(temp_audio)
                # Store speech segments for recursive call
                self._cached_speech_segments = speech_segments
            else:
                print("Using cached speech segments...")
                speech_segments = getattr(self, '_cached_speech_segments', [])
                if not speech_segments:
                    speech_segments = self.detect_speech_segments(temp_audio)
            
            if not speech_segments:
                print("Warning: No speech detected in audio")
                return False
            
            # Calculate sync offset - now passing the audio path
            print("Calculating sync offset...")
            offset = self.calculate_sync_offset(subtitles, speech_segments, temp_audio)
            
            # Case 1: Zero Timestamps Reconstructed
            if offset == "ZERO_TIMESTAMPS_RECONSTRUCTED":
                print("Timestamps reconstructed from speech analysis")
                
                # Write the reconstructed subtitles first
                self.write_subtitle_file(subtitles, self.output_path)
                print(f"Initial reconstruction saved to: {self.output_path}")
                
                # Now run synchronization again on the reconstructed file for fine-tuning
                print("\n" + "="*60)
                print("PHASE 2: Fine-tuning the reconstructed timestamps...")
                print("="*60)
                
                # Create a new synchronizer instance for the reconstructed file
                fine_tune_synchronizer = SubtitleSynchronizer(
                    self.video_path, 
                    self.output_path,  # Use the reconstructed file as input
                    self.output_path   # Same output path
                )
                
                # Copy cached speech segments to avoid re-detection
                fine_tune_synchronizer._cached_speech_segments = speech_segments
                
                # Mark as recursive call to prevent infinite reconstruction
                fine_tune_synchronizer._is_recursive_call = True
                
                # Run synchronization recursively
                fine_tune_result = fine_tune_synchronizer.synchronize(is_recursive_call=True)
                
                if fine_tune_result:
                    print("✅ Two-phase synchronization completed successfully!")
                    print(f"Final synchronized subtitles saved to: {self.output_path}")
                else:
                    print("✅ Initial reconstruction was already well-aligned!")
                
                return True
            else:
                if is_recursive_call:
                    print(f"Fine-tuning offset: {offset:.2f} seconds")
                else:
                    print(f"Calculated offset: {offset:.2f} seconds")
                
                # Case 2: Normal Offset Calculated
                if abs(offset) >= 0.05:  # 50ms threshold instead of 100ms
                    synced_subtitles = self.apply_sync_offset(subtitles, offset)
                    self.write_subtitle_file(synced_subtitles, self.output_path)
                    if is_recursive_call:
                        print(f"Fine-tuned subtitles saved to: {self.output_path}")
                    else:
                        print(f"Synchronized subtitles saved to: {self.output_path}")
                    return True
                else:
                    if is_recursive_call:
                        print("Fine-tuning complete - subtitles are well synchronized")
                    # Case 3: Already Synchronized
                    else:
                        print("Subtitles are already well synchronized")
                    return False
        finally:
            # Clean up temporary audio file only if this is not a recursive call
            if not is_recursive_call and os.path.exists(temp_audio):
                os.remove(temp_audio)
    
    def apply_sync_offset(self, subtitles, offset):
        """Apply sync offset to subtitles"""
        synced_subtitles = []
        
        for sub in subtitles:
            synced_sub = sub.copy()
            synced_sub['start'] = max(0, sub['start'] + offset)
            synced_sub['end'] = max(synced_sub['start'], sub['end'] + offset)
            synced_subtitles.append(synced_sub)
        
        return synced_subtitles
    
    # STEP 7: Output Generation
    #-------------------------------------
    def write_subtitle_file(self, subtitles, output_path):
        """Write subtitles to file in original format"""
        if not subtitles:
            return
        
        format_type = subtitles[0]['format']
        
        with open(output_path, 'w', encoding='utf-8') as f:
            if format_type == 'vtt':
                f.write("WEBVTT\n\n")
                for sub in subtitles:
                    start_time = self._seconds_to_vtt_time(sub['start'])
                    end_time = self._seconds_to_vtt_time(sub['end'])
                    f.write(f"{start_time} --> {end_time}\n")
                    f.write(f"{sub['text']}\n\n")
            
            elif format_type == 'srt':
                for sub in subtitles:
                    start_time = self._seconds_to_srt_time(sub['start'])
                    end_time = self._seconds_to_srt_time(sub['end'])
                    f.write(f"{sub['index']}\n")
                    f.write(f"{start_time} --> {end_time}\n")
                    f.write(f"{sub['text']}\n\n")

    def _seconds_to_srt_time(self, seconds):
        """Convert seconds to SRT time format"""
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        secs = int(seconds % 60)
        millisecs = int((seconds % 1) * 1000)
        return f"{hours:02d}:{minutes:02d}:{secs:02d},{millisecs:03d}"

    def _seconds_to_vtt_time(self, seconds):
        """Convert seconds to VTT time format"""
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        secs = int(seconds % 60)
        millisecs = int((seconds % 1) * 1000)
        return f"{hours:02d}:{minutes:02d}:{secs:02d}.{millisecs:03d}"

def main():

    video_file = "inputs/2mins.mp4"
    subtitle_file = "inputs/2mins.srt"
    
    synchronizer = SubtitleSynchronizer(video_file, subtitle_file)
    synchronizer.synchronize()

if __name__ == "__main__":
    main()