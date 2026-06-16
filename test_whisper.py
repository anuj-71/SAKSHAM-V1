import logging
import time

logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')

logging.info("Importing Faster-Whisper...")
try:
    from faster_whisper import WhisperModel
    logging.info("Import successful. Loading model...")
    start_t = time.time()
    
    # Try default compute_type first to see if int8 was the issue
    model = WhisperModel("base", device="cpu", compute_type="default")
    
    elapsed = time.time() - start_t
    logging.info(f"Model loaded successfully in {elapsed:.2f}s!")
except Exception as e:
    logging.error(f"Failed: {e}")
    import traceback
    traceback.print_exc()
