"""
Faster-Whisper Test Script.

This script tests the faster-whisper implementation to ensure it works correctly
before integrating it into the main application.
"""
import asyncio
import time
from faster_whisper_manager import faster_whisper_model_manager
from faster_whisper_service import faster_whisper_transcribe
import audio_processing_utils
from logging_config import get_logger

logger = get_logger(__name__)


async def test_faster_whisper():
    """Test the faster-whisper implementation."""
    print("🚀 Testing Faster-Whisper Implementation...")

    try:
        # Initialize models
        print("1. Initializing faster-whisper models...")
        success = await faster_whisper_model_manager.initialize_models(warm_up=True)

        if not success:
            print("❌ Failed to initialize models")
            return False

        print("✅ Models initialized successfully")

        # Get stats
        stats = faster_whisper_model_manager.get_stats()
        print(f"📊 Stats: {stats}")

        # Create a test audio file
        print("2. Creating test audio file...")
        test_audio = audio_processing_utils.create_dummy_audio_file(
            "test_faster_whisper.wav")
        print(f"✅ Test audio created: {test_audio}")

        # Test transcription
        print("3. Testing transcription...")
        start_time = time.time()

        transcribed_text, detected_language = await faster_whisper_transcribe(
            test_audio, force_language="en"
        )

        end_time = time.time()
        duration = end_time - start_time

        print(f"✅ Transcription completed in {duration:.2f}s")
        print(f"📝 Text: '{transcribed_text}'")
        print(f"🌐 Language: {detected_language}")

        # Clean up
        print("4. Cleaning up...")
        if audio_processing_utils.safe_remove_file(test_audio):
            print("✅ Test file cleaned up")

        # Final stats
        final_stats = faster_whisper_model_manager.get_stats()
        print(f"📊 Final stats: {final_stats}")

        print("🎉 Faster-Whisper test completed successfully!")
        return True

    except Exception as e:
        logger.error("❌ Test failed: %s", str(e))
        print(f"❌ Test failed: {e}")
        return False


async def benchmark_comparison():
    """Benchmark faster-whisper performance."""
    print("\n🏁 Running Performance Benchmark...")

    try:
        # Create test audio
        test_audio = audio_processing_utils.create_dummy_audio_file(
            "benchmark_test.wav")

        # Test faster-whisper
        print("Testing faster-whisper performance...")
        start_time = time.time()
        fw_text, fw_lang = await faster_whisper_transcribe(test_audio)
        fw_duration = time.time() - start_time

        print(f"⚡ Faster-Whisper: {fw_duration:.2f}s")
        print(f"   Text: '{fw_text[:50]}...' Language: {fw_lang}")

        # Clean up
        audio_processing_utils.safe_remove_file(test_audio)

        return True

    except Exception as e:
        logger.error("❌ Benchmark failed: %s", str(e))
        return False


if __name__ == "__main__":
    print("🔧 Faster-Whisper Test Suite")
    print("=" * 50)

    async def main():
        # Run basic test
        test_success = await test_faster_whisper()

        if test_success:
            # Run benchmark
            await benchmark_comparison()

        print("\n" + "=" * 50)
        print("🏁 Test suite completed")

    asyncio.run(main())
