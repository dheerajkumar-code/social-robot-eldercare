#!/usr/bin/env python3
"""
Module 2 — ASR Node (v2)
--------------------------
Real-time speech-to-text using Faster-Whisper with energy-based VAD.

What this fixes vs asr_node.py (v1):
  - v1: Fixed 3s timer → Whisper hallucinates "Thank you." on silence
  - v2: Energy VAD → only transcribe when real speech detected
  - v1: Blocks ROS2 executor thread during 2-3s transcription
  - v2: Transcription runs in background thread, ROS2 never blocked
  - v1: 'small' model (244M params, ~3s latency on Pi5)
  - v2: 'base' model default (74M params, ~1s latency on Pi5)
  - v1: No no_speech_prob check → hallucinations published
  - v2: Rejects segments with no_speech_prob > 0.6
  - v1: Transcribes silence between words as separate utterances
  - v2: VAD state machine captures complete utterances end-to-end

VAD State Machine:
  SILENCE → (RMS > threshold) → ONSET
  ONSET   → (held N blocks)   → SPEAKING
  SPEAKING→ (RMS < threshold, held M blocks) → TRANSCRIBE
  TRANSCRIBE → send to Whisper thread → back to SILENCE

Topics published:
  /asr_text     std_msgs/String  — transcribed text (non-empty, real speech only)
  /asr_status   std_msgs/String  — "listening" | "speaking" | "transcribing"

Standalone:
  python3 asr_node_v2.py --device 2

ROS2:
  python3 asr_node_v2.py --ros --device 2

Pi5 recommended:
  python3 asr_node_v2.py --ros --model base --device 2
"""

import os
import sys
import time
import queue
import logging
import threading
import argparse
import numpy as np
import sounddevice as sd

# Optional ROS2
try:
    import rclpy
    from rclpy.node import Node
    from std_msgs.msg import String
    ROS_AVAILABLE = True
except ImportError:
    ROS_AVAILABLE = False

try:
    from faster_whisper import WhisperModel
except ImportError:
    raise ImportError("Install faster-whisper: pip install faster-whisper")

# ─────────────────────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────────────────────
SAMPLE_RATE    = 16000
BLOCK_SIZE     = 1600          # 100ms per block (SAMPLE_RATE * 0.1)
LANGUAGE       = "en"
DEFAULT_MODEL  = "base"        # base=74M params, ~1s on Pi5 | small=244M ~3s

# VAD parameters — tune these for your environment
ENERGY_THRESHOLD    = 0.01     # RMS above this = speech (lower = more sensitive)
ONSET_BLOCKS        = 3        # consecutive loud blocks before SPEAKING (300ms)
TAIL_SILENCE_BLOCKS = 15       # consecutive quiet blocks after speech ends (1.5s)
MAX_SPEECH_SECONDS  = 12.0     # max utterance length before forced transcription
MIN_SPEECH_SECONDS  = 0.4      # minimum utterance length (reject shorter)

# Whisper quality filters
MAX_NO_SPEECH_PROB  = 0.60     # reject segment if Whisper thinks it is silence
MIN_AVG_LOG_PROB    = -1.2     # reject very low confidence transcriptions

LOG_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "data", "logs")
os.makedirs(LOG_DIR, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(os.path.join(LOG_DIR, "asr_transcripts.log")),
    ],
)
logger = logging.getLogger("asr_v2")


# ─────────────────────────────────────────────────────────────
# VAD State Machine
# ─────────────────────────────────────────────────────────────

class VADState:
    SILENCE    = "silence"
    ONSET      = "onset"
    SPEAKING   = "speaking"
    TRANSCRIBE = "transcribe"


class EnergyVAD:
    """
    Block-level energy VAD.
    Feed audio blocks one at a time via process_block().
    When a complete utterance is detected, returns it via get_utterance().
    """

    def __init__(self,
                 sample_rate      = SAMPLE_RATE,
                 energy_threshold = ENERGY_THRESHOLD,
                 onset_blocks     = ONSET_BLOCKS,
                 tail_blocks      = TAIL_SILENCE_BLOCKS,
                 max_seconds      = MAX_SPEECH_SECONDS,
                 min_seconds      = MIN_SPEECH_SECONDS):

        self.sr               = sample_rate
        self.threshold        = energy_threshold
        self.onset_blocks     = onset_blocks
        self.tail_blocks      = tail_blocks
        self.max_samples      = int(max_seconds * sample_rate)
        self.min_samples      = int(min_seconds * sample_rate)

        self.state            = VADState.SILENCE
        self.speech_buffer    = []   # raw float32 samples during speaking
        self.onset_counter    = 0
        self.tail_counter     = 0
        self._utterance_queue = queue.Queue()

    def process_block(self, block: np.ndarray):
        """
        Feed one audio block. Internally updates state machine.
        Call get_utterance() to retrieve completed utterances.
        """
        rms = float(np.sqrt(np.mean(block ** 2)))

        if self.state == VADState.SILENCE:
            if rms > self.threshold:
                self.onset_counter += 1
                self.speech_buffer.append(block.copy())
                if self.onset_counter >= self.onset_blocks:
                    self.state = VADState.SPEAKING
                    logger.debug("VAD → SPEAKING")
            else:
                self.onset_counter = 0
                self.speech_buffer = []

        elif self.state == VADState.SPEAKING:
            self.speech_buffer.append(block.copy())

            if rms < self.threshold:
                self.tail_counter += 1
                if self.tail_counter >= self.tail_blocks:
                    self._finalize_utterance()
            else:
                self.tail_counter = 0

            # Force transcription if utterance too long
            total_samples = sum(b.size for b in self.speech_buffer)
            if total_samples >= self.max_samples:
                logger.debug("VAD → forced TRANSCRIBE (max length)")
                self._finalize_utterance()

    def _finalize_utterance(self):
        audio = np.concatenate(self.speech_buffer).astype(np.float32)
        self.speech_buffer  = []
        self.onset_counter  = 0
        self.tail_counter   = 0
        self.state          = VADState.SILENCE

        if audio.size >= self.min_samples:
            self._utterance_queue.put(audio)
            logger.debug(f"VAD → utterance ready ({audio.size/self.sr:.2f}s)")
        else:
            logger.debug(f"VAD → utterance too short ({audio.size/self.sr:.2f}s), discarded")

    def get_utterance(self, block=False, timeout=0.05):
        """Returns next completed utterance (float32 array) or None."""
        try:
            return self._utterance_queue.get(block=block, timeout=timeout)
        except queue.Empty:
            return None

    @property
    def current_state(self) -> str:
        return self.state


# ─────────────────────────────────────────────────────────────
# ASR Engine
# ─────────────────────────────────────────────────────────────

class ASREngine:
    """
    Wraps Faster-Whisper with quality filtering.
    Transcription runs in a background thread — never blocks callers.
    """

    def __init__(self,
                 model_size: str   = DEFAULT_MODEL,
                 on_result=None):
        """
        Args:
            model_size : "tiny" | "base" | "small"
            on_result  : callback(text: str) called from transcription thread
        """
        self.on_result  = on_result
        self._work_q    = queue.Queue()
        self._stop_flag = threading.Event()

        # Load model
        model_dir = os.path.join(os.path.dirname(__file__), "models", model_size)
        os.makedirs(model_dir, exist_ok=True)

        logger.info(f"Loading Faster-Whisper '{model_size}' (int8, CPU)...")
        self.model = WhisperModel(
            model_size,
            device="cpu",
            compute_type="int8",
            download_root=model_dir,       # cache locally — no HuggingFace on every restart
        )
        logger.info("✅ Model loaded.")

        # Start background transcription thread
        self._thread = threading.Thread(target=self._worker, daemon=True)
        self._thread.start()

    def submit(self, audio: np.ndarray):
        """Non-blocking — add audio to transcription queue."""
        self._work_q.put(audio)

    def _worker(self):
        """Background thread: drains work queue and calls Whisper."""
        while not self._stop_flag.is_set():
            try:
                audio = self._work_q.get(timeout=0.1)
            except queue.Empty:
                continue

            text = self._transcribe(audio)
            if text and self.on_result:
                self.on_result(text)

    def _transcribe(self, audio: np.ndarray) -> str:
        """
        Run Whisper on audio array.
        Returns text string or None if rejected.
        """
        try:
            t0 = time.time()
            segments, info = self.model.transcribe(
                audio,
                language=LANGUAGE,
                beam_size=1,
                best_of=1,
                vad_filter=True,               # Silero VAD inside Whisper
                vad_parameters={
                    "min_silence_duration_ms": 300,
                    "speech_pad_ms": 100,
                },
                condition_on_previous_text=False,  # prevent hallucination loops
            )

            # Check no_speech_prob on whole audio
            if info.all_language_probs is not None:
                pass  # language detected fine

            # Collect segments
            texts = []
            for seg in segments:
                # Per-segment quality filters
                if seg.no_speech_prob > MAX_NO_SPEECH_PROB:
                    logger.debug(f"Rejected (no_speech_prob={seg.no_speech_prob:.2f}): {seg.text!r}")
                    continue
                if seg.avg_logprob < MIN_AVG_LOG_PROB:
                    logger.debug(f"Rejected (low logprob={seg.avg_logprob:.2f}): {seg.text!r}")
                    continue

                t = seg.text.strip()
                if t:
                    texts.append(t)

            latency = time.time() - t0
            audio_dur = audio.size / SAMPLE_RATE

            if texts:
                result = " ".join(texts)
                logger.info(f"[ASR] '{result}'  (audio={audio_dur:.1f}s, latency={latency:.2f}s)")
                return result
            else:
                logger.debug(f"[ASR] Rejected (no valid segments) audio={audio_dur:.1f}s")
                return None

        except Exception as e:
            logger.error(f"Transcription error: {e}")
            return None

    def stop(self):
        self._stop_flag.set()


# ─────────────────────────────────────────────────────────────
# Standalone runner
# ─────────────────────────────────────────────────────────────

def run_standalone(device=None, model_size=DEFAULT_MODEL):
    print("=" * 55)
    print("  Module 2 — ASR v2 (VAD + Faster-Whisper)")
    print("=" * 55)

    vad = EnergyVAD()
    results = []

    def on_result(text):
        print(f"\n🗣️  {text}\n")
        results.append(text)

    asr = ASREngine(model_size=model_size, on_result=on_result)
    raw_q = queue.Queue()

    def audio_cb(indata, frames, time_info, status):
        if status:
            logger.warning(f"Audio status: {status}")
        raw_q.put(indata[:, 0].copy() if indata.ndim > 1 else indata.copy().flatten())

    print(f"\n🎙️  Listening on device {device}... (Ctrl+C to stop)\n")
    print(f"    VAD threshold : {ENERGY_THRESHOLD}")
    print(f"    Model         : {model_size}")
    print(f"    Language      : {LANGUAGE}\n")

    with sd.InputStream(device=device, channels=1, samplerate=SAMPLE_RATE,
                        blocksize=BLOCK_SIZE, dtype="float32", callback=audio_cb):
        try:
            while True:
                try:
                    block = raw_q.get(timeout=0.2)
                except queue.Empty:
                    continue

                vad.process_block(block)

                # Status indicator
                state = vad.current_state
                if state == VADState.SPEAKING:
                    print("🔴 Speaking...", end="\r")
                else:
                    print("⚪ Listening  ", end="\r")

                # Pull completed utterances
                utterance = vad.get_utterance()
                if utterance is not None:
                    print("🔵 Transcribing...", end="\r")
                    asr.submit(utterance)

        except KeyboardInterrupt:
            print("\n\nStopped.")
            asr.stop()


# ─────────────────────────────────────────────────────────────
# ROS2 Node
# ─────────────────────────────────────────────────────────────

class ASRNodeV2(Node):
    """
    ROS2 node for ASR with VAD.

    Publishes:
      /asr_text    std_msgs/String — transcribed text
      /asr_status  std_msgs/String — "listening" | "speaking" | "transcribing"

    The ROS2 executor thread is NEVER blocked by transcription.
    Whisper runs in its own daemon thread (ASREngine._worker).
    """

    def __init__(self, device=None, model_size=DEFAULT_MODEL):
        super().__init__("asr_node_v2")

        # Publishers
        self.pub_text   = self.create_publisher(String, "/asr_text",   10)
        self.pub_status = self.create_publisher(String, "/asr_status", 10)

        # ROS2 parameters
        self.declare_parameter("energy_threshold", ENERGY_THRESHOLD)
        self.declare_parameter("model_size",       model_size)
        self.declare_parameter("language",         LANGUAGE)

        # VAD
        self.vad = EnergyVAD(
            energy_threshold=self.get_parameter("energy_threshold").value
        )

        # ASR engine with result callback
        self.asr = ASREngine(
            model_size=self.get_parameter("model_size").value,
            on_result=self._on_transcription,
        )

        # Audio input queue
        self._raw_q = queue.Queue(maxsize=200)

        # Audio stream
        self._device = device
        self._stream = sd.InputStream(
            device=device,
            channels=1,
            samplerate=SAMPLE_RATE,
            blocksize=BLOCK_SIZE,
            dtype="float32",
            callback=self._audio_cb,
        )
        self._stream.start()

        # Timer to drain VAD (runs in ROS2 timer thread — very fast, no blocking)
        self.create_timer(0.05, self._drain_vad)   # 50ms poll

        self.get_logger().info(
            f"ASR node v2 started | model={model_size} | "
            f"VAD threshold={ENERGY_THRESHOLD}"
        )

    def _audio_cb(self, indata, frames, time_info, status):
        """Audio callback — just push raw blocks to queue. Never blocks."""
        if status:
            self.get_logger().warning(f"Audio status: {status}")
        block = indata[:, 0].copy() if indata.ndim > 1 else indata.flatten()
        try:
            self._raw_q.put_nowait(block)
        except queue.Full:
            pass  # drop oldest if overflowing

    def _drain_vad(self):
        """
        Called by ROS2 timer every 50ms.
        Drains raw audio queue, feeds VAD, submits utterances to ASR.
        This is fast — no Whisper calls happen here.
        """
        # Drain raw audio queue into VAD
        while not self._raw_q.empty():
            try:
                block = self._raw_q.get_nowait()
                self.vad.process_block(block)
            except queue.Empty:
                break

        # Publish current VAD state
        self._publish_status(self.vad.current_state)

        # Check for completed utterances
        utterance = self.vad.get_utterance()
        if utterance is not None:
            self._publish_status("transcribing")
            self.asr.submit(utterance)  # non-blocking — goes to background thread

    def _on_transcription(self, text: str):
        """Called from ASR background thread when transcription is ready."""
        msg = String()
        msg.data = text
        self.pub_text.publish(msg)
        self.get_logger().info(f"Published: '{text}'")

    def _publish_status(self, status: str):
        msg = String()
        msg.data = status
        self.pub_status.publish(msg)

    def destroy_node(self):
        self._stream.stop()
        self._stream.close()
        self.asr.stop()
        super().destroy_node()


# ─────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────

def list_devices():
    print("\nAvailable audio input devices:\n")
    for i, d in enumerate(sd.query_devices()):
        if d["max_input_channels"] > 0:
            print(f"  {i}: {d['name']} (inputs: {d['max_input_channels']})")


def parse_args():
    p = argparse.ArgumentParser(description="ASR node v2 (VAD + Faster-Whisper)")
    p.add_argument("--ros",           action="store_true",          help="Run as ROS2 node")
    p.add_argument("--device",        type=int,   default=None,     help="Audio device index")
    p.add_argument("--model",         type=str,   default=DEFAULT_MODEL,
                   choices=["tiny", "base", "small"],               help="Whisper model size")
    p.add_argument("--threshold",     type=float, default=ENERGY_THRESHOLD,
                                                                     help="VAD energy threshold")
    p.add_argument("--list-devices",  action="store_true",          help="List audio devices")
    return p.parse_args()


def main():
    args = parse_args()

    if args.list_devices:
        list_devices()
        return

    # Automatically intercept device 0 since it is an HDMI output
    if args.device == 0:
        print("\n" + "="*60)
        print(" ⚠️  WARNING: Device 0 is an HDMI output, not a microphone!")
        print("    Automatically switching to your default microphone.")
        print("="*60 + "\n")
        args.device = None

    if args.ros:
        if not ROS_AVAILABLE:
            logger.error("rclpy not available. Install ROS2 Humble.")
            sys.exit(1)
        rclpy.init()
        node = ASRNodeV2(device=args.device, model_size=args.model)
        try:
            rclpy.spin(node)
        except KeyboardInterrupt:
            node.get_logger().info("Shutting down ASR node v2")
        finally:
            node.destroy_node()
            rclpy.shutdown()
    else:
        run_standalone(device=args.device, model_size=args.model)


if __name__ == "__main__":
    main()
