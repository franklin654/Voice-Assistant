import time
import numpy as np
import whisper
from pvrecorder import PvRecorder
import torch
import logging
import re
from chat_model import graph, get_config
from langchain.messages import HumanMessage

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
logger.debug(f"Available Recording Devices: {PvRecorder.get_available_devices()}")


def main():
    # --- Configuration ---
    MODEL_SIZE = "turbo"    # Options: tiny, base, small, medium, large-v2
    FRAME_LEGNTH = 512      # 512 samples @ 16kHz ~= 32ms (Standard for Silero VAD)
    SAMPLE_RATE = 16000     # 16kHz is required by both Silero and Whisper
    SILENCE_THRESHOLD = 2.0 # Seconds of silence to trigger transcription
    VAD_THRESHOLD = 0.5     # Confidence threshold for "is speech" (0.0 to 1.0)

    # --- 1. Load Models ---
    logger.info(f"Loading Whisper ({MODEL_SIZE})...")
    # Run on GPU with "cuda" or CPU with "cpu" (and float32)
    whisper_model = whisper.load_model(MODEL_SIZE, device="cuda" if torch.cuda.is_available() else "cpu")

    logger.info("Loading Silero VAD...")
    # Load Silero VAD from PyTorch Hub
    vad_model, utils = torch.hub.load(repo_or_dir="snakers4/silero-vad", 
                                      model="silero_vad",
                                      force_reload=False) # type: ignore
    (get_speech_timestamps, save_audio, read_audio, VADIterator, collect_chunks) = utils

    # --- 2. Setup Recorder ---
    recorder = PvRecorder(device_index=-1, frame_length=FRAME_LEGNTH)
    logger.debug(f"Recording from: {recorder.selected_device}")

    # buffers
    audio_buffer = []       # Raw INT16 audio frames
    speech_detected = False
    silence_start_time = None

    try:
        recorder.start()

        while True:
            # A. Read Frame
            frame = recorder.read()
            # Append to buffer immediatly to avoid cutting off start of words
            audio_buffer.extend(frame)

            # B. Prepare Frame for VAD (Convert int16 -> float32)
            # Silero expects normalized float32 tensor
            frame_np = np.array(frame, dtype=np.int16)
            frame_float = frame_np.astype(np.float32) / 32768.0
            frame_tensor = torch.from_numpy(frame_float)

            # C. Check Voice Activity
            # We check just this frame's probability
            speech_prob = vad_model(frame_tensor, SAMPLE_RATE).item()

            is_speech = speech_prob > VAD_THRESHOLD

            if is_speech:
                if not speech_detected:
                    print("[+] Speech started...")
                    speech_detected = True
                silence_start_time = None # Reset silence timer
            
            # D. Handle Silence & Trigger Transcription
            elif speech_detected:
                # If we were previously speaking, check how long we've been silent
                if silence_start_time is None:
                    silence_start_time = time.time()

                # If silence exceeds threshold, transcribe!
                elif (time.time() - silence_start_time) > SILENCE_THRESHOLD:
                    print("[+] Silence detected. Transcribing...")

                    # 1. Convert complete buffer to float32 numpy array for Whisper
                    full_audio_data = np.array(audio_buffer, dtype=np.int16)
                    audio_float32 = full_audio_data.astype(np.float32) / 32768.0

                    # 2. Transcribe
                    result = whisper_model.transcribe(audio_float32)

                    # 3. Print Results
                    print(f"[Transcribed]: {result["text"]}")

                    response = graph.invoke({"messages": HumanMessage(result["text"])}, config=get_config())
                    response["messages"][-1].pretty_print()
                    
                    if re.search("bye", result["text"], flags=re.IGNORECASE) != None: # type: ignore
                        break
                    
                    # 4. Reset
                    audio_buffer.clear()
                    speech_detected = False
                    silence_start_time = None
                    print("[+] Listening...")

            # Keep buffer size manageable if no speech ever starts
            if not speech_detected and len(audio_buffer) > SAMPLE_RATE * 30:
                # Trim buffer to last 1 second to keep context but free memory
                audio_buffer = audio_buffer[-(SAMPLE_RATE):]

    except KeyboardInterrupt:
        print("\nStopping...")
    finally:
        recorder.delete()
    

if __name__ == "__main__":
    main()