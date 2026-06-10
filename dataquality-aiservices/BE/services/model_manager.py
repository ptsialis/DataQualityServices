"""
Global LLM Model Manager

This module manages a singleton instance of the LLM model,
ensuring it is loaded only once at application startup and
reused throughout the entire application lifecycle.
"""

import logging
import threading
import time

logger = logging.getLogger(__name__)

# Global model instance
_model_instance = None
_model_lock = threading.Lock()
_model_loading = False
_model_ready = threading.Event()


def initialize_model(model_name: str = "Qwen/Qwen3-4B-Instruct-2507-FP8", quantization: str = '4bit'):
    """
    Initialize the global LLM model instance in a background thread.
    
    This function returns immediately and starts loading the model asynchronously.
    The Flask app continues to accept requests while the model loads in the background.
    
    Args:
        model_name (str): HuggingFace model identifier (default: "Qwen/Qwen3-4B-Instruct-2507-FP8")
        quantization (str): Quantization mode - '8bit', '4bit', or '' for full precision
    """
    global _model_instance, _model_loading
    
    with _model_lock:
        if _model_instance is not None:
            logger.info("✓ Model already initialized, skipping initialization")
            _model_ready.set()
            return
        
        if _model_loading:
            logger.info("⏳ Model loading already in progress, skipping duplicate initialization")
            return
        
        _model_loading = True
    
    logger.info(f"🚀 Starting background model loading thread for {model_name}...")
    
    # Start loading in background thread
    def _load_model_background():
        global _model_instance
        try:
            logger.info(f"🔄 Background thread: Initializing LLM model: {model_name}")
            logger.info(f"🔄 Background thread: Quantization mode: {quantization}")
            
            try:
                from src.services_featuretype_personal.huggingface_model import load_model
                logger.info("✓ Successfully imported load_model from huggingface_model")
            except ImportError as ie:
                logger.error(f"✗ Failed to import load_model: {str(ie)}")
                raise
            
            logger.info("⏳ Loading model weights (this may take 5-15 minutes)...")
            _model_instance = load_model(model=model_name, quantization=quantization)
            
            logger.info("✓ ✓ ✓ LLM model successfully loaded and ready for use! ✓ ✓ ✓")
            _model_ready.set()
            
        except ImportError as e:
            logger.error(f"✗ Import error during model loading: {str(e)}")
            logger.error(f"  Make sure src/services_featuretype_personal/huggingface_model.py exists")
            _model_ready.set()
        except Exception as e:
            logger.error(f"✗ Exception during model loading: {type(e).__name__}: {str(e)}")
            import traceback
            logger.error(traceback.format_exc())
            _model_ready.set()
    
    # Start thread with daemon=False so it continues even if main thread exits
    thread = threading.Thread(target=_load_model_background, daemon=False, name="LLMModelLoader")
    thread.start()
    logger.info("✓ Background thread started successfully")


def get_model(wait_timeout: int = 600):
    """
    Get the global LLM model instance.
    
    If model is still loading, this will wait up to wait_timeout seconds for it to be ready.
    
    Args:
        wait_timeout (int): Maximum seconds to wait for model to load (default: 600 = 10 minutes)
    
    Returns:
        The loaded model pipeline instance
        
    Raises:
        RuntimeError: If model loading failed or timeout exceeded
    """
    global _model_instance
    
    # If model is already loaded, return immediately
    if _model_instance is not None:
        logger.info("✓ Returning already-loaded model instance")
        return _model_instance
    
    # Wait for model to be ready (with timeout)
    logger.info(f"⏳ Waiting for model to load (timeout: {wait_timeout}s)...")
    if not _model_ready.wait(timeout=wait_timeout):
        logger.error(f"✗ Model loading timed out after {wait_timeout} seconds")
        raise RuntimeError(
            f"Model loading timed out after {wait_timeout} seconds. "
            "The model is still loading in the background. Please try again in a few moments."
        )
    
    # Check if model loaded successfully
    if _model_instance is None:
        logger.error("✗ Model is None even after ready signal - loading must have failed")
        raise RuntimeError(
            "Model failed to load during initialization. "
            "Check server logs for error details."
        )
    
    logger.info("✓ Model is ready!")
    return _model_instance


def is_model_initialized() -> bool:
    """
    Check if the global LLM model has been initialized (without waiting).
    
    Returns:
        True if model is fully initialized and ready, False if still loading or failed
    """
    global _model_instance
    return _model_instance is not None


def reset_model():
    """
    Reset the global model instance (useful for testing or reloading).
    
    Warning: This will unload the model and free its memory.
    """
    global _model_instance, _model_loading, _model_ready
    _model_instance = None
    _model_loading = False
    _model_ready.clear()
    logger.info("Model instance has been reset")
