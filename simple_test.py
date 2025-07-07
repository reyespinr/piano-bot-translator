"""
Simple faster-whisper test to verify basic functionality.
"""
import os
import tempfile
import numpy as np
from faster_whisper import WhisperModel
import soundfile as sf


def create_test_audio():
    """Create a simple test audio file."""
    # Create 3 seconds of test audio with a simple sine wave
    duration = 3.0
    sample_rate = 16000
    frequency = 440.0  # A4 note

    t = np.linspace(0, duration, int(sample_rate * duration))
    audio = np.sin(2 * np.pi * frequency * t) * 0.3

    # Save to temporary file
    temp_file = tempfile.NamedTemporaryFile(suffix='.wav', delete=False)
    sf.write(temp_file.name, audio, sample_rate)
    temp_file.close()

    return temp_file.name


def test_basic_model():
    """Test basic model loading and transcription."""
    try:
        print("🚀 Testing basic faster-whisper functionality...")

        # Create test audio
        print("1. Creating test audio...")
        audio_file = create_test_audio()
        print(f"   ✅ Created: {audio_file}")

        # Load model with auto compute type for CUDA
        print("2. Loading model...")
        model = WhisperModel("base", device="cuda", compute_type="auto")
        print("   ✅ Model loaded successfully")

        # Test transcription
        print("3. Testing transcription...")
        segments, info = model.transcribe(audio_file, beam_size=1)
        segments_list = list(segments)

        print(f"   ✅ Transcription completed")
        print(f"   📊 Detected language: {info.language}")
        print(f"   📊 Language probability: {info.language_probability:.2f}")
        print(f"   📊 Segments: {len(segments_list)}")

        # Clean up
        os.unlink(audio_file)
        print("   ✅ Cleaned up test file")

        print("🎉 Basic test completed successfully!")
        return True

    except Exception as e:
        print(f"❌ Basic test failed: {e}")
        if 'audio_file' in locals():
            try:
                os.unlink(audio_file)
            except:
                pass
        return False


if __name__ == "__main__":
    test_basic_model()
