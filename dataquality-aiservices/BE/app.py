import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="sklearn.base")

import os
import logging
from flask import Flask
from flask_session import Session
from flask_cors import CORS
from routes.main_routes import bp as main_bp
from services.session_state import init_session_state
from services.model_manager import initialize_model


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def create_app():
    warnings.filterwarnings("ignore", category=UserWarning, module="sklearn.base")

    app = Flask(__name__)

    # --------------------------------------------------------------
    # CORS
    # --------------------------------------------------------------
    cors_origins = [
        origin.strip()
        for origin in os.environ.get(
            "CORS_ORIGINS",
            "http://localhost:5173",
        ).split(",")
        if origin.strip()
    ]

    CORS(
        app,
        resources={r"/*": {"origins": cors_origins}},
        supports_credentials=True,
    )

    # --------------------------------------------------------------
    # Server-side Flask sessions
    # --------------------------------------------------------------
    secret_key = os.environ.get("SECRET_KEY")

    if not secret_key:
        raise RuntimeError(
            "Missing SECRET_KEY environment variable. "
            "Set a long random value before starting the backend."
        )

    app.config.update(
        SECRET_KEY=secret_key,
        SESSION_TYPE="filesystem",
        SESSION_FILE_DIR=os.environ.get(
            "SESSION_FILE_DIR",
            os.path.join(os.getcwd(), "Session_Flask"),
        ),
        SESSION_PERMANENT=False,
        SESSION_USE_SIGNER=False,   
        SESSION_COOKIE_NAME="my_session",
        SESSION_COOKIE_HTTPONLY=True,
        SESSION_COOKIE_SAMESITE=os.environ.get(
            "SESSION_COOKIE_SAMESITE",
            "Lax",
        ),
        SESSION_COOKIE_SECURE=(
            os.environ.get("SESSION_COOKIE_SECURE", "false").lower() == "true"
        ),
    )
    

    os.makedirs(app.config["SESSION_FILE_DIR"], mode=0o700, exist_ok=True)

    Session(app)

    # --------------------------------------------------------------
    # Routes
    # --------------------------------------------------------------
    app.register_blueprint(main_bp)

    @app.before_request
    def before_request():
        init_session_state()

    # --------------------------------------------------------------
    # Background LLM loading
    #
    # Important:
    # Keep one application worker until the LLM is moved into a
    # dedicated inference service or GPU worker.
    # --------------------------------------------------------------
    try:
        logger.info("")
        logger.info("=" * 70)
        logger.info("🚀 APPLICATION STARTUP - INITIALIZING LLM MODEL")
        logger.info("=" * 70)

        initialize_model(
            model_name="Qwen/Qwen3-4B",
            quantization="4bit",
        )

        logger.info("=" * 70)
        logger.info("✓ Flask is ready to accept requests!")
        logger.info("⏳ LLM model is loading in the background...")
        logger.info("=" * 70)
        logger.info("")

    except Exception as exc:
        logger.error("=" * 70)
        logger.error(f"✗ Error during model initialization startup: {exc}")
        logger.error("=" * 70)

        import traceback
        logger.error(traceback.format_exc())

    return app


if __name__ == "__main__":
    app = create_app()

    app.run(
        debug=True,
        host="0.0.0.0",
        port=5000,
    )