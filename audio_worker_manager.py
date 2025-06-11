"""
Background worker management for audio transcription_service.
Handles worker threads, health monitoring, and queue management.
"""
import asyncio
import threading
import queue
import time
import os
from typing import Optional, Callable, Awaitable
from logging_config import get_logger

logger = get_logger(__name__)


class AudioWorkerManager:
    """Manages transcription worker threads with health monitoring."""

    def __init__(self, num_workers: int, event_loop: Optional[asyncio.AbstractEventLoop] = None):
        self.num_workers = num_workers
        self.event_loop = event_loop
        self.queue = queue.Queue(maxsize=50)
        self.running = True
        self.workers = []
        self.timer = None

        # Health monitoring
        self.worker_health = {}
        self.last_health_check = 0
        self.worker_restart_count = {}
        self.max_restarts_per_worker = 3

        # Callback for transcription processing
        self.transcription_callback: Optional[Callable] = None

    def set_transcription_callback(self, callback: Callable):
        """Set the callback function for processing transcriptions."""
        self.transcription_callback = callback

    def start_workers(self):
        """Initialize and start worker threads."""
        for i in range(self.num_workers):
            worker_id = f"TranscriptionWorker-{i+1}"
            thread = threading.Thread(
                target=self._transcription_worker,
                args=(worker_id,),
                daemon=True,
                name=worker_id
            )
            thread.start()
            # Initialize worker health tracking
            self.workers.append(thread)
            self.mark_worker_healthy(worker_id)
            self.worker_restart_count[worker_id] = 0
            logger.info("Started transcription worker %d/%d (%s)",
                        i+1, self.num_workers, worker_id)

    def queue_audio_file(self, user, audio_filename, timeout=1.0):
        """Queue an audio file for transcription_service."""
        try:
            self.queue.put((user, audio_filename), timeout=timeout)
            logger.debug("Queued audio file for user %s: %s",
                         user, audio_filename)
            return True
        except queue.Full:
            logger.warning(
                "Transcription queue full, dropping audio for user %s", user)
            # Clean up the audio file
            if audio_filename and os.path.exists(audio_filename):
                try:
                    os.remove(audio_filename)
                except Exception:
                    pass
            return False

    def get_queue_size(self):
        """Get current queue size."""
        return self.queue.qsize()

    def manage_queue_overflow(self):
        """Manage queue overflow by dropping oldest items when queue is too full."""
        queue_size = self.queue.qsize()
        max_queue_size = 35  # Lower than maxsize=50 to prevent total overflow

        if queue_size > max_queue_size:
            dropped_count = 0
            # Drop up to 10 oldest items to make room
            max_drops = min(10, queue_size - 25)  # Target queue size of 25

            logger.warning("Queue overflow detected (%d items), dropping %d oldest items",
                           queue_size, max_drops)

            for _ in range(max_drops):
                try:
                    user, audio_file = self.queue.get_nowait()
                    # Clean up the dropped audio file
                    if audio_file and os.path.exists(audio_file):
                        try:
                            os.remove(audio_file)
                            logger.debug(
                                "Cleaned up dropped audio file: %s", audio_file)
                        except Exception:
                            pass
                    dropped_count += 1
                    self.queue.task_done()
                except queue.Empty:
                    break

            if dropped_count > 0:
                logger.warning(
                    "Dropped %d audio files due to queue overflow", dropped_count)

        return queue_size

    def mark_worker_healthy(self, worker_id: str):
        """Mark a worker as healthy with current timestamp."""
        self.worker_health[worker_id] = time.time()

    def is_worker_healthy(self, worker_id: str, timeout: float = 60.0) -> bool:
        """Check if a worker is healthy (responded within timeout)."""
        last_response = self.worker_health.get(worker_id, 0)
        return (time.time() - last_response) < timeout

    def get_unhealthy_workers(self, timeout: float = 60.0) -> list:
        """Get list of worker IDs that haven't responded within timeout."""
        current_time = time.time()
        unhealthy = []
        for worker_id, last_response in self.worker_health.items():
            if (current_time - last_response) > timeout:
                unhealthy.append(worker_id)
        return unhealthy

    def check_health(self):
        """Check health of all worker threads and restart failed ones."""
        current_time = time.time()

        # Check health more frequently for better crash detection
        if (current_time - self.last_health_check) < 15.0:  # Reduced from 30s to 15s
            return

        self.last_health_check = current_time
        logger.debug("Performing enhanced worker health check...")

        # Check for unhealthy workers (reduced timeout for faster detection)
        unhealthy_workers = self.get_unhealthy_workers(
            timeout=45.0)  # Reduced from 90s
        dead_workers = []

        # Check for dead threads
        for i, worker in enumerate(self.workers):
            worker_id = worker.name
            if not worker.is_alive():
                dead_workers.append((i, worker_id))
                logger.warning(
                    "⚠️ Worker thread %s is dead - will restart", worker_id)

        # CRITICAL: Also check for workers that might be stuck
        stuck_workers = []
        queue_size = self.queue.qsize()

        # If queue is backing up and workers aren't responding, they might be stuck
        if queue_size > 20:
            for worker_id in self.worker_health.keys():
                last_health = self.worker_health.get(worker_id, 0)
                if (current_time - last_health) > 30.0:  # Haven't checked in for 30s with high queue
                    stuck_workers.append((None, worker_id))
                    logger.warning("⚠️ Worker %s appears stuck (queue size: %d) - will restart",
                                   worker_id, queue_size)

        # Restart dead, unresponsive, or stuck workers
        workers_to_restart = dead_workers + \
            [(None, w) for w in unhealthy_workers if w not in [d[1] for d in dead_workers]] + \
            stuck_workers

        if workers_to_restart:
            logger.warning("🔧 Found %d workers that need restart (dead: %d, unhealthy: %d, stuck: %d)",
                           len(workers_to_restart), len(dead_workers), len(unhealthy_workers), len(stuck_workers))

        for worker_index, worker_id in workers_to_restart:
            restart_count = self.worker_restart_count.get(worker_id, 0)

            if restart_count >= self.max_restarts_per_worker:
                logger.error("❌ Worker %s has exceeded max restart attempts (%d), not restarting",
                             worker_id, self.max_restarts_per_worker)
                continue

            logger.warning(
                "🔄 Restarting problematic worker: %s (restart #%d)", worker_id, restart_count + 1)

            # Create new worker thread
            new_thread = threading.Thread(
                target=self._transcription_worker,
                args=(worker_id,),
                daemon=True,
                name=worker_id
            )
            new_thread.start()

            # Update worker list
            if worker_index is not None:
                self.workers[worker_index] = new_thread
            else:
                # Find and replace the worker by name
                for i, w in enumerate(self.workers):
                    if w.name == worker_id:
                        self.workers[i] = new_thread
                        break

            # Reset health tracking
            self.mark_worker_healthy(worker_id)
            self.worker_restart_count[worker_id] = restart_count + 1

            logger.info("✅ Successfully restarted worker: %s", worker_id)

    def _transcription_worker(self, worker_id):
        """Background worker to process transcription queue with robust error handling and crash recovery."""
        thread_name = threading.current_thread().name
        logger.info("%s started with crash recovery", thread_name)

        consecutive_failures = 0
        max_consecutive_failures = 5
        processing_count = 0

        # CRITICAL FIX: Create a proper event loop for this worker thread
        loop = None
        try:
            # CRITICAL FIX: Don't try to get existing loop, always create new one
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            logger.debug("%s created new event loop", thread_name)
        except Exception as e:
            logger.error("%s failed to create event loop: %s",
                         thread_name, str(e))
            return

        # Mark worker as healthy initially
        if worker_id:
            self.mark_worker_healthy(worker_id)

        while self.running:
            audio_file = None
            user = None
            queue_item_acquired = False

            try:
                # Get item from queue with timeout
                user, audio_file = self.queue.get(timeout=1.0)
                queue_item_acquired = True

                logger.debug("%s processing: %s for user %s (count: %d)",
                             thread_name, audio_file, user, processing_count + 1)

                # CRITICAL: Pre-validate audio file exists before processing
                if not audio_file or not os.path.exists(audio_file):
                    logger.warning(
                        "%s skipping missing audio file: %s", thread_name, audio_file)
                    continue

                # Process the transcription with timeout protection
                try:
                    if self.transcription_callback:
                        # CRITICAL: Add timeout to prevent hanging on transcription
                        transcription_task = self.transcription_callback(
                            audio_file, user)
                        await_result = asyncio.wait_for(
                            transcription_task, timeout=60.0)
                        loop.run_until_complete(await_result)

                    # Successful processing - reset failure counter
                    consecutive_failures = 0
                    processing_count += 1

                    # Mark worker as healthy after successful processing
                    if worker_id:
                        self.mark_worker_healthy(worker_id)

                    logger.debug("%s successfully processed item %d",
                                 thread_name, processing_count)

                except asyncio.TimeoutError:
                    logger.error("%s transcription timeout for file %s (user %s)",
                                 thread_name, audio_file, user)
                    consecutive_failures += 1
                    # Clean up the hung audio file
                    if audio_file and os.path.exists(audio_file):
                        try:
                            os.remove(audio_file)
                            logger.warning(
                                "%s cleaned up timeout audio file: %s", thread_name, audio_file)
                        except Exception:
                            pass

                except Exception as transcription_error:
                    logger.error("%s transcription error for %s: %s",
                                 thread_name, audio_file, str(transcription_error))
                    consecutive_failures += 1
                    # Clean up the failed audio file
                    if audio_file and os.path.exists(audio_file):
                        try:
                            os.remove(audio_file)
                            logger.warning(
                                "%s cleaned up failed audio file: %s", thread_name, audio_file)
                        except Exception:
                            pass

            except queue.Empty:
                # No work available - periodic health check
                if worker_id:
                    self.mark_worker_healthy(worker_id)
                continue

            except Exception as queue_error:
                logger.error("%s critical error in queue processing: %s",
                             thread_name, str(queue_error))
                consecutive_failures += 1

                # CRITICAL: If we can't even process the queue, something is very wrong
                if consecutive_failures >= max_consecutive_failures:
                    logger.error("%s experienced %d consecutive failures, exiting to trigger restart",
                                 thread_name, consecutive_failures)
                    break

            finally:
                # CRITICAL: Always mark task as done if we acquired a queue item
                if queue_item_acquired:
                    try:
                        self.queue.task_done()
                    except Exception as task_done_error:
                        logger.error("%s error marking task done: %s",
                                     thread_name, str(task_done_error))

        # Worker is exiting - log final statistics
        logger.warning("%s exiting after processing %d items (consecutive failures: %d)",
                       thread_name, processing_count, consecutive_failures)

        # CRITICAL FIX: Properly close the event loop when worker exits
        try:
            if loop and not loop.is_closed():
                loop.close()
                logger.debug("%s closed event loop", thread_name)
        except Exception as e:
            logger.error("%s error closing event loop: %s",
                         thread_name, str(e))

        logger.info("%s exiting", thread_name)

    def cleanup(self):
        """Clean up all workers and resources."""
        logger.info("Starting worker manager cleanup process...")

        # Set running flag to False
        logger.debug("Setting worker running flag to False")
        self.running = False

        # Cancel timer if it exists
        if self.timer:
            try:
                self.timer.cancel()
            except:
                pass

        # Add a small delay to let threads terminate naturally
        logger.debug("Waiting for threads to terminate naturally")
        time.sleep(0.5)  # Increased from 0.2 to 0.5 for more threads

        # Clear the queue to unblock any waiting threads
        try:
            while not self.queue.empty():
                try:
                    self.queue.get_nowait()
                    self.queue.task_done()
                except:
                    break
        except (ValueError, RuntimeError) as e:
            logger.debug("Queue clear error: %s", str(e))

        # Wait for all worker threads to finish with timeout
        logger.debug("Waiting for worker threads to join...")
        for i, worker in enumerate(self.workers):
            try:
                worker.join(timeout=2.0)
                logger.debug("Worker thread %d successfully joined", i + 1)
            except:
                logger.warning("Worker thread %d failed to join", i + 1)

        logger.info(
            "Cleaning up worker manager resources - %d workers processed", self.num_workers)

        # Print final thread status
        alive_threads = sum(1 for worker in self.workers if worker.is_alive())
        if alive_threads > 0:
            logger.warning(
                "%d worker threads still alive after cleanup", alive_threads)
        else:
            logger.info("All worker threads successfully terminated")
