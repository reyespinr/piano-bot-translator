"""
Utility script to clean up stray WAV files.

This script removes any temporary WAV files that might have been left behind
by the application during unexpected shutdowns or errors.
"""
import os
import time
import re
import subprocess
import sys
from logging_config import get_logger

# Initialize logger for this module
logger = get_logger(__name__)


def clean_temp_files(directory='.', pattern=r'\d+_\d+_speech\.wav$', age_minutes=0):
    """
    Clean up temporary speech files in the specified directory.

    Args:
        directory (str): Directory to clean
        pattern (str): Regex pattern to match temporary files
        age_minutes (float): Remove files older than this many minutes (0 for all)
    """
    now = time.time()
    count = 0
    size = 0
    failed_files = []

    logger.info("Scanning for temporary audio files...")

    # Compile the regex pattern
    file_pattern = re.compile(pattern)

    for filename in os.listdir(directory):
        if file_pattern.search(filename):
            filepath = os.path.join(directory, filename)

            try:
                # Check file age if age_minutes > 0
                if age_minutes > 0:
                    file_age = now - os.path.getmtime(filepath)
                    if file_age < (age_minutes * 60):  # Convert minutes to seconds
                        continue  # Skip files that are too new

                # Get file size before deleting
                file_size = os.path.getsize(filepath)

                # Try to delete the file
                if force_delete_file(filepath):
                    count += 1
                    size += file_size
                else:
                    failed_files.append(filename)

            except (OSError, PermissionError) as e:
                logger.error("Error processing %s: %s", filename, e)
                failed_files.append(filename)

    if count > 0:
        logger.info(
            "Cleanup: Removed %d temporary audio files (%.2f MB)",
            count, size/1024/1024)
    else:
        logger.info("No temporary audio files found to clean up.")

    if failed_files:
        logger.warning("Failed to remove %d files: %s%s",
                       len(failed_files),
                       ', '.join(failed_files[:5]),
                       " and %d more" % (len(failed_files)-5) if len(failed_files) > 5 else "")


def force_delete_file(filepath):
    """Try multiple methods to delete a file."""
    # Try standard deletion first
    try:
        os.remove(filepath)
        return True
    except (PermissionError, OSError):
        pass

    # Try with Windows commands if on Windows
    if sys.platform == "win32":
        try:
            subprocess.run(f'del /F "{filepath}"', shell=True, check=False)
            if not os.path.exists(filepath):
                return True
        except (subprocess.SubprocessError, OSError):
            pass

    # Try again after a brief pause
    try:
        time.sleep(0.5)
        os.remove(filepath)
        return True
    except (PermissionError, OSError):
        return False


if __name__ == "__main__":
    clean_temp_files()
