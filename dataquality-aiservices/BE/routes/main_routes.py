from flask import Blueprint, request, jsonify, Response, session, send_file
import pandas as pd
import json
import math
import io
import os
import base64
import random
from services.session_state import (
    save_dataframe_to_session,
    load_dataframe_from_session,
    get_session_file_path,
    get_workspace_path,
    remove_current_workspace,
)
from services.model_manager import get_model, is_model_initialized
import src.global_controller as gc
import src.meta_data_export as md
import src.summary as sm
import src.plots as plts
from src.services_featuretype_personal.services_LLM import (
    execute_featuretype_LLM,
    execute_personal_llm,
    execute_dataset_llm,
)
from src.services_featuretype_personal.services_LLM import FeaturizeFile, Load_RF, FeatureExtraction
from src.file_handler_random import serialize_df_for_llm, load_dataset_into_dataframe
from src.file_security import validate_uploaded_file, sanitize_dataframe, log_security_event
from plotly.utils import PlotlyJSONEncoder
from datetime import datetime
import zipfile
import src.random_forest_train as rftrain
import subprocess, tempfile
import shutil
import src.xgboost_train as xgbtrain
import uuid
import pickle
import re
from pathlib import Path


MAX_COLUMNS_PER_LLM_CALL = 20
MAX_SAMPLES_PER_COLUMN = 6
bp = Blueprint('main', __name__)


#wandelt DataFrame in Dictionary um
def _feature_types_result_df_to_mapping(df):

    if df is None or df.empty:
        raise ValueError("feature types dataframe is empty")

    cols = {c.lower(): c for c in df.columns}

    attr_col = None
    pred_col = None

    for cand in ["attribute_name", "column", "feature", "variable", "column_name"]:
        if cand in cols:
            attr_col = cols[cand]
            break

    for cand in ["prediction", "predicted_type", "type"]:
        if cand in cols:
            pred_col = cols[cand]
            break

    if attr_col is None or pred_col is None:
        raise ValueError(f"Could not detect feature-type columns in {list(df.columns)}")

    #ergebnis als Dictionary aufbauen
    out = {}
    for _, row in df[[attr_col, pred_col]].dropna().iterrows():
        out[str(row[attr_col])] = str(row[pred_col])

    return out


#wandelt DataFrame in Dictionary um
def _personal_result_df_to_mapping(df):

    if df is None or df.empty:
        raise ValueError("personal dataframe is empty")

    cols = {c.lower(): c for c in df.columns}

    attr_col = None
    pred_col = None

    for cand in ["column", "attribute_name", "feature", "variable", "column_name"]:
        if cand in cols:
            attr_col = cols[cand]
            break

    for cand in ["prediction", "predicted_type", "type"]:
        if cand in cols:
            pred_col = cols[cand]
            break

    if attr_col is None or pred_col is None:
        raise ValueError(f"Could not detect personal-data columns in {list(df.columns)}")

    #ergebnis als Dictionary aufbauen
    out = {}
    for _, row in df[[attr_col, pred_col]].dropna().iterrows():
        out[str(row[attr_col])] = str(row[pred_col])

    return out

#wandelt DataFrame in zusammenfassung um
def _anomaly_df_to_summary(df):

    if df is None or df.empty:
        raise ValueError("anomaly dataframe is empty")

    if "Anomaly" not in df.columns:
        raise ValueError("anomaly dataframe has no 'Anomaly' column")

    s = pd.to_numeric(df["Anomaly"], errors="coerce").fillna(0).astype(int)
    anomaly_idx = df.index[s == 1].tolist()

    return {
        "column_name": "Anomaly",
        "totalRows": int(len(df)),
        "anomalies_total": int(len(anomaly_idx)),
        "anomalies_percent": float((len(anomaly_idx) / len(df)) * 100.0 if len(df) else 0.0),
        "anomaly_indices": [int(i) for i in anomaly_idx],
    }


# Model artifacts are stored inside the current session workspace.
def _artifact_path(filename: str) -> str:
    if not filename or filename != os.path.basename(filename):
        raise ValueError("Invalid artifact filename")

    return str(
        get_session_file_path("artifacts", filename)
    )


def df_to_json(df: pd.DataFrame) -> dict:
    """Hilfsfunktion: Wandelt DataFrame in JSON-kompatibles Dict mit columns + data um."""
    if df is None or df.empty:
        return {"columns": [], "data": []}
    df_clean = df.where(pd.notnull(df), None)
    return {
        "columns": df_clean.columns.tolist(),
        "data": df_clean.to_dict(orient="records")
    }
    
    

@bp.route('/status', methods=['GET'])
def status():
    """Check if the LLM model is ready for use."""
    try:
        is_ready = is_model_initialized()
        
        if is_ready:
            status_msg = "✓ LLM model is fully loaded and ready!"
            status_val = "ready"
        else:
            status_msg = "⏳ LLM model is still loading in the background..."
            status_val = "loading"
        
        return jsonify({
            "status": status_val,
            "llm_ready": is_ready,
            "message": status_msg,
            "timestamp": datetime.now().isoformat()
        }), 200
    except Exception as e:
        return jsonify({
            "status": "error",
            "llm_ready": False,
            "message": f"Error checking model status: {str(e)}",
            "timestamp": datetime.now().isoformat()
        }), 500

def _load_file_by_format(file) -> tuple[pd.DataFrame, str]:
    """
    Load file based on format with security validation.
    Supports CSV, ARFF, JSON, Parquet, Arrow, and more.
    Returns (DataFrame, file_extension)
    """
    filename = file.filename.lower()
    ext = Path(filename).suffix[1:].lower()  # Remove leading dot
    
    
    # Save temporary upload inside the current session workspace.
    upload_dir = get_workspace_path() / "uploads"
    upload_dir.mkdir(mode=0o700, parents=True, exist_ok=True)

    fd, temp_path = tempfile.mkstemp(
        prefix="upload_",
        suffix=f".{ext}" if ext else "",
        dir=upload_dir,
    )

    os.close(fd)
    file.save(temp_path)
    
    try:
        # Security validation
        is_valid, error_msg, warnings, _ = validate_uploaded_file(
            temp_path,
            file.filename,
            ext
        )
        
        if not is_valid:
            log_security_event("FILE_VALIDATION_FAILED", file.filename, error_msg)
            raise ValueError(f"Sicherheitsprüfung fehlgeschlagen: {error_msg}")
        
        if warnings:
            log_security_event("FILE_SECURITY_WARNINGS", file.filename, f"{len(warnings)} warnings")
            print(f">>> ⚠ Security warnings for {file.filename}:")
            for warning in warnings:
                print(f"    - {warning}")
        
        # Try loading with universal handler (now with proper extension)
        df = load_dataset_into_dataframe(temp_path, max_rows=None)
        
        # Sanitize DataFrame to remove injection attempts
        df_safe, sanitize_warnings = sanitize_dataframe(df)
        
        if sanitize_warnings:
            log_security_event("DATA_SANITIZATION", file.filename, f"{len(sanitize_warnings)} cells sanitized")
            print(f">>> 🛡️ Sanitized {len(sanitize_warnings)} potentially malicious cells")
        
        return df_safe, ext
    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)


@bp.route('/upload', methods=['POST'])
def upload_data_beginning():
    file = request.files.get('file')
    if not file:
        return jsonify({"success": False, "error": "Keine Datei hochgeladen."}), 400

    try:
        df, file_ext = _load_file_by_format(file)
        df_original = df.copy(deep=True)
    except ValueError as e:
        # Security validation failed
        log_security_event("UPLOAD_REJECTED", file.filename, str(e))
        return jsonify({
            "success": False, 
            "error": str(e)
        }), 400
    except Exception as e:
        log_security_event("FILE_LOAD_ERROR", file.filename, str(e))
        return jsonify({
            "success": False, 
            "error": f"Fehler beim Einlesen der Datei: {str(e)}"
        }), 400

    target_variable = request.form.get('target')
    if target_variable not in df.columns:
        return jsonify({"success": False, "error": "Bitte Zielvariable auswählen."}), 400

    gc.add_metadata("Target variable", target_variable)
    feature_typ_inference = gc.infer_problem_type(df[target_variable])
    gc.add_metadata("User has chosen", feature_typ_inference)
    
    use_llm_feature_type_inference = request.form.get("use_llm_feature_type_inference", "false") == "true"
    use_llm_personal_data_detection = request.form.get("use_llm_personal_data_detection", "false") == "true"

    print(">>> LLM Feature Type Inference:", use_llm_feature_type_inference)
    print(">>> LLM Personalized Data Detection:", use_llm_personal_data_detection)

    model_llm = None
    if use_llm_feature_type_inference or use_llm_personal_data_detection:
        try:
            print(">>> 🔍 Attempting to get LLM model...")
            model_llm = get_model()  # This will wait for the model with timeout
            print(">>> ✓ Successfully obtained global LLM model")
        except RuntimeError as e:
            error_msg = str(e)
            print(f">>> ✗ RuntimeError getting model: {error_msg}")
            # If model is not ready, we should retry later
            # Return 503 Service Unavailable to indicate temp failure
            return jsonify({
                "success": False, 
                "error": f"LLM model is not ready yet. {error_msg} Please wait a moment and try again.",
                "retry": True,
                "retryAfter": 30
            }), 503  # 503 Service Unavailable
        except Exception as e:
            error_msg = str(e)
            print(f">>> ✗ Unexpected error getting model: {type(e).__name__}: {error_msg}")
            return jsonify({
                "success": False, 
                "error": f"Unexpected error with LLM model: {error_msg}",
                "retry": False
            }), 500

    if (
        use_llm_feature_type_inference
        and use_llm_personal_data_detection
    ):
        (
            feature_typ_inference,
            personal_0,
            personal_1,
        ) = execute_dataset_llm(
            df_original,
            model_llm,
        )

    elif use_llm_feature_type_inference:
        feature_typ_inference = execute_featuretype_LLM(
            df_original,
            model_llm,
        )

        personal_0, personal_1 = (
            gc.make_prediciton_personal(df_original)
        )

    elif use_llm_personal_data_detection:
        feature_typ_inference = gc.make_predictions(
            df,
            "RandomForest",
        )

        personal_0, personal_1 = execute_personal_llm(
            df_original,
            model_llm,
        )

    else:
        feature_typ_inference = gc.make_predictions(
            df,
            "RandomForest",
        )

        personal_0, personal_1 = (
            gc.make_prediciton_personal(df_original)
        )

    time_or_tab = gc.check_time_series(
        df,
        feature_typ_inference,
    )

    gc.add_metadata(
        "Time or Tab",
        time_or_tab,
    )



    imputed_df, _ = gc.impute_mean_mode(df.copy(deep=True))


    if time_or_tab == "Tabular":
        df_anomalies, threshold = gc.detect_anomalies(df)
    else:
        _, normalized_data, feats = gc.load_and_preprocess_data(df, feature_typ_inference)
        anomalies, _, _, threshold = gc.detect_anomalies_tranad(normalized_data, feats, 95)
        df_anomalies = df.copy()
        df_anomalies["Anomaly"] = anomalies

    # Serialize DataFrame with LLM-optimized format if LLM is being used
    serialized_data = None
    if use_llm_feature_type_inference or use_llm_personal_data_detection:
        try:
            print(">>> 📦 Serializing data with LLM-optimized format...")
            serialized_data = serialize_df_for_llm(
                df_original,
                max_rows=50,
                include_markdown=True,
                include_html=True
            )
            print(">>> ✓ Data serialized successfully")
        except Exception as e:
            print(f">>> ⚠ Warning: Serialization failed: {e}")
            # Don't fail the upload, just continue without serialized data

    session["data"] = {
        "State": True,
        "threshold": threshold,
        "time_or_tab": time_or_tab,
        "name": file.filename,
        "target_variable": target_variable,
        "file_format": file_ext,
        "use_llm_feature_type_inference": use_llm_feature_type_inference,
        "use_llm_personal_data_detection": use_llm_personal_data_detection,
    }
    
    save_dataframe_to_session("original", df_original)
    save_dataframe_to_session("impute", imputed_df)
    save_dataframe_to_session("anomaly", df_anomalies)
    save_dataframe_to_session("inference", feature_typ_inference[["prediction","Attribute_name"]])
    save_dataframe_to_session("personal_0", personal_0)
    save_dataframe_to_session("personal_1", personal_1)

    if serialized_data:
        serialized_path = get_session_file_path(
            "serialized",
            "llm_input.json",
        )

        with open(serialized_path, "w", encoding="utf-8") as file:
            json.dump(
                serialized_data,
                file,
                ensure_ascii=False,
                default=str,
            )

        session["serialized_data_path"] = str(serialized_path)
    else:
        session.pop("serialized_data_path", None)

    session["starting_process"] = False
    session["intermediate_process"] = True
    
    session.modified = True

    return jsonify({
        "success": True,
        "message": "Upload erfolgreich verarbeitet.",
    })


@bp.route('/original')
def show_orig_data():
    df = load_dataframe_from_session("original")
    desc = sm.describe_dataframe(df)
    fig = plts.plot_boxplots(df)

    return Response(
        json.dumps({
            "dataframe": df_to_json(df),
            "description": df_to_json(desc),
            "boxplot": fig.to_plotly_json()
        }, default=str),
        mimetype='application/json'
    )
    
    
@bp.route('/datagraphs')
def show_datagraphs():
    df = load_dataframe_from_session("original")

    sub = request.args.get("sub") #boxplots | histograms | correlationMatrix | featureImportance

    payload = {
        "boxplot": None,
        "histograms": None,
        "correlation_matrix": None,
        "feature_importance": None
    }

    if sub in (None, "", "boxplots"):
        boxplot_fig = plts.plot_boxplots(df)
        payload["boxplot"] = boxplot_fig.to_plotly_json()

    elif sub == "histograms":
        figs = plts.plot_histograms(df)
        payload["histograms"] = [fig.to_plotly_json() for fig in figs]

    elif sub == "correlationMatrix":
        fig = plts.plot_correlation_matrix(df)
        payload["correlation_matrix"] = fig.to_plotly_json() if fig else None

    elif sub == "featureImportance":
        imputed = load_dataframe_from_session("impute")
        inference = load_dataframe_from_session("inference")
        target = session.get("data", {}).get("target_variable", "")

        if imputed is None or imputed.empty or not target:
            payload["feature_importance"] = None
        else:
            df_encoded = gc.one_hot_encode_columns(
                imputed,
                inference,
                target_col=target
            )
            fig = plts.plot_feature_importance(df_encoded, target, top_n=20)
            payload["feature_importance"] = fig.to_plotly_json() if fig is not None else None
            
    return Response(
        json.dumps(payload, cls=PlotlyJSONEncoder),
        mimetype="application/json"
    )


@bp.route('/inference')
def show_inference():
    inference = load_dataframe_from_session("inference")
    original = load_dataframe_from_session("original")
    target = session.get("data", {}).get("target_variable", "")

    explanation_fig = sm.explanation_feature_type_inference(original, target)
    explanation_json = explanation_fig.to_plotly_json()
    
    return Response(
        json.dumps({
            "dataframe": df_to_json(inference),
            "target_variable": target,
            "explanation_plot": explanation_json
        }, default=str),
        mimetype='application/json'
    )


@bp.route('/imputation')
def show_imputation():
    original = load_dataframe_from_session("original")
    imputed = load_dataframe_from_session("impute")
    inference = load_dataframe_from_session("inference")

    explanation_fig = sm.explanation_imputation(original, inference)
    try:
        explanation_json = explanation_fig.to_plotly_json()
    except AttributeError:
        explanation_json = explanation_fig

    return Response(
        json.dumps({
            "original": df_to_json(original),
            "imputed": df_to_json(imputed),
            "explanation_plot": explanation_json
        }, default=str),
        mimetype='application/json'
    )


@bp.route('/anomaly')
def show_anomalies():
    df = load_dataframe_from_session("anomaly")
    threshold = session.get("data", {}).get("threshold", 0)
    time_or_tab = session.get("data", {}).get("time_or_tab", "")

    explanation_fig = sm.explanation_anomaly(threshold, time_or_tab)

    try:
        explanation_json = explanation_fig.to_plotly_json()
    except AttributeError:
        if isinstance(explanation_fig, float) and math.isnan(explanation_fig):
            explanation_json = None
        else:
            explanation_json = explanation_fig

    return Response(
        json.dumps({
            "anomalies": df_to_json(df),
            "threshold": threshold,
            "time_or_tab": time_or_tab,
            "explanation_plot": explanation_json
        }, default=str),
        mimetype='application/json'
    )


@bp.route('/personal')
def show_personal():
    personal_1 = load_dataframe_from_session("personal_1")
    personal_0 = load_dataframe_from_session("personal_0")

    return jsonify({
        "personal_1": df_to_json(personal_1),
        "personal_0": df_to_json(personal_0)
    })


@bp.route('/metadata')
def show_meta_data():
    original = load_dataframe_from_session("original")
    inference = load_dataframe_from_session("inference")
    personal_1 = load_dataframe_from_session("personal_1")
    anomaly = load_dataframe_from_session("anomaly")

    metadata, featwise = gc.extract_metadata(original, inference)
    demo_meta = sm.meta_data_from_demos(inference, original, anomaly, personal_1)

    if isinstance(metadata, dict):
        metadata = pd.DataFrame.from_dict(metadata, orient="index")
    if isinstance(featwise, dict):
        featwise = pd.DataFrame.from_dict(featwise, orient="index")
    if isinstance(demo_meta, dict):
        demo_meta = pd.DataFrame.from_dict(demo_meta, orient="index")

    final_df = gc.metadata_to_dataframe(metadata.where(pd.notnull(metadata), None))
    final_df.to_csv(
    get_session_file_path("exports", "m_test.csv"),
    index=False,
)

    name = session.get("data", {}).get("name", "export")
    md.generate_metadata_json(original, name.split(".")[0])

    session["starting_process"] = False
    session["intermediate_process"] = False
    session["ending_process"] = True
    session["show_upload"] = True

    return Response(
        json.dumps({
            "general_metadata": df_to_json(metadata),
            "featurewise_metadata": df_to_json(featwise),
            "demo_metadata": df_to_json(demo_meta)
        }, default=str),
        mimetype="application/json"
    )


@bp.route('/summary')
def show_summary():
    inference = load_dataframe_from_session("inference")
    anomaly = load_dataframe_from_session("anomaly")
    original = load_dataframe_from_session("original")
    personal_0 = load_dataframe_from_session("personal_0")

    plots = {
        "inference": plts.plot_inference_bar(inference).to_plotly_json(),
        "anomaly": plts.plot_anomaly_pie(anomaly).to_plotly_json(),
        "imputation": plts.plot_imputation_bar(original, inference).to_plotly_json(),
        "personal": plts.plot_personal_pie(personal_0).to_plotly_json()
    }

    return Response(json.dumps(plots, default=str), mimetype='application/json')


@bp.route("/detectedCounts")
def get_detected_counts():
    def extract_count(value):
        """
        Converts values like:
        - 2
        - "2"
        - "2 Different Types Detected"
        into integer 2.
        """
        if value is None:
            return 0

        if isinstance(value, (int, float)):
            return int(value)

        value_str = str(value).strip()
        match = re.search(r"\d+", value_str)

        if match:
            return int(match.group())

        return 0

    original = load_dataframe_from_session("original")
    imputed = load_dataframe_from_session("impute")
    inference = load_dataframe_from_session("inference")
    anomaly = load_dataframe_from_session("anomaly")
    personal_0 = load_dataframe_from_session("personal_0")

    counts = {}

    if inference is not None and not inference.empty and "prediction" in inference.columns:
        counts["inference"] = extract_count(sm.summary_inference(inference))

    if original is not None and not original.empty and imputed is not None and not imputed.empty:
        replaced = (original.isna() & imputed.notna()).sum().sum()
        counts["imputation"] = int(replaced)
    elif original is not None and not original.empty:
        counts["imputation"] = extract_count(sm.summary_imputation(original))

    if anomaly is not None and not anomaly.empty and "Anomaly" in anomaly.columns:
        counts["anomaly"] = extract_count(sm.summary_anomaly(anomaly))

    if personal_0 is not None and not personal_0.empty:
        counts["personal"] = extract_count(sm.summary_personal(personal_0))

    return jsonify(counts)



@bp.route('/cleaned')
def show_clean_data():
    original_df = load_dataframe_from_session("original")
    imputed = load_dataframe_from_session("impute")
    inference = load_dataframe_from_session("inference")
    anomaly = load_dataframe_from_session("anomaly")
    personal_1 = load_dataframe_from_session("personal_1")

    target = session.get("data", {}).get("target_variable", None)

    if imputed is None or imputed.empty:
        return Response(
            json.dumps({"error": "Imputed DataFrame ist leer/nicht vorhanden."}),
            mimetype="application/json",
            status=400,
        )

    if inference is None or inference.empty:
        return Response(
            json.dumps({"error": "Inference DataFrame ist leer/nicht vorhanden."}),
            mimetype="application/json",
            status=400,
        )

    final_df = gc.one_hot_encode_columns(
        imputed,
        inference,
        df_anomalies=anomaly,
        df_personal=personal_1,
        target_col=target,
    )
    
    original_df.to_csv(
    get_session_file_path("exports", "original_df.csv"),
    index=False,
)

    final_df.to_csv(
        get_session_file_path("exports", "final_df.csv"),
        index=False,
    )

    return Response(
        json.dumps({
            "original": df_to_json(original_df),
            "final": df_to_json(final_df)
        }, default=str),
        mimetype='application/json'
    )
    
@bp.route("/model/download/<filename>", methods=["GET"])
def download_model_artifact(filename):

    try:
        file_path = _artifact_path(filename)
    except Exception:
        return jsonify({"success": False, "error": "Invalid filename"}), 400

    if not os.path.isfile(file_path):
        return jsonify({"success": False, "error": "Artifact not found"}), 404

    return send_file(
        file_path,
        as_attachment=True,
        download_name=os.path.basename(file_path),
        mimetype="application/octet-stream",
    )


def _write_table_csv_to_zip(zf, name: str, df):

    if df is None:
        return

    csv_bytes = df.to_csv(index=False).encode("utf-8")
    zf.writestr(name, csv_bytes)


def _build_model_zip_artifact(zip_path: str, model_obj, result_obj: dict, split_data: dict, model_filename: str):
    os.makedirs(os.path.dirname(zip_path), exist_ok=True)

    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        # Modell als Pickle ins ZIP
        model_bytes = pickle.dumps(model_obj)
        zf.writestr(model_filename, model_bytes)

        # Ergebnis-JSON ins ZIP
        zf.writestr(
            "result.json",
            json.dumps(result_obj, ensure_ascii=False, indent=2).encode("utf-8")
        )

        #Split-table in ZIP
        if split_data:
            _write_table_csv_to_zip(zf, "splits/Training_Data.csv", split_data.get("train_table"))
            _write_table_csv_to_zip(zf, "splits/Validation_Data.csv", split_data.get("validation_table"))
            _write_table_csv_to_zip(zf, "splits/Test_Data.csv", split_data.get("test_table"))

def _append_split_tables_to_existing_zip(zip_path: str, split_data: dict):

    if not split_data:
        return

    with zipfile.ZipFile(zip_path, "a", compression=zipfile.ZIP_DEFLATED) as zf:
        _write_table_csv_to_zip(zf, "splits/Training_Data.csv", split_data.get("train_table"))
        _write_table_csv_to_zip(zf, "splits/Validation_Data.csv", split_data.get("validation_table"))
        _write_table_csv_to_zip(zf, "splits/Test_Data.csv", split_data.get("test_table"))

@bp.route("/model", methods=["GET", "POST"])
def model_endpoint():
    if request.method == "GET":
        meta = session.get("data", {}) or {}
        return jsonify({
            "success": True,
            "message": "Model endpoint ready.",
            "has_session": bool(meta),
            "target_variable": meta.get("target_variable", ""),
            "time_or_tab": meta.get("time_or_tab", ""),
            "dataset_name": meta.get("name", ""),
        })


    req = request.get_json(silent=True) or {}

    model_key = (req.get("model_key") or "").strip()
    mode = (req.get("search_mode") or "none").strip()
    use_cv = bool(req.get("cross_validation", False))
    time_limit_s = int(req.get("time_limit_s") or 60)
    seed_raw = req.get("seed", None)
    if seed_raw not in (None, ""):
        seed = int(seed_raw)
    else:
        #TRUE/FALSE für RANDOM/FIXED SEED / aus Testzwecken aktuell auf FALSE!!!
        USE_RANDOM_SEED = False

        if USE_RANDOM_SEED:
            seed = random.randint(1, 10_000_000)
        else:
            seed = 1337

    if model_key not in ("rf", "xgb", "automl"):
        return jsonify({"success": False, "error": "Only 'rf', 'xgb' and 'automl' are implemented right now."}), 400

    #Target Variable aus Session
    meta = session.get("data", {}) or {}
    target = meta.get("target_variable", "") or (req.get("target_variable") or "")
    if not target:
        return jsonify({"success": False, "error": "Missing target_variable (session or request)."}), 400

    #DataFrame aus Session
    imputed = load_dataframe_from_session("impute")
    if imputed is None or getattr(imputed, "empty", True):
        return jsonify({"success": False, "error": "Imputed DataFrame missing/empty."}), 400


    #RF/XGB one-hot / numerisch

    if model_key in ("rf", "xgb"):
        inference = load_dataframe_from_session("inference")
        anomaly   = load_dataframe_from_session("anomaly")
        personal  = load_dataframe_from_session("personal_1")

        if inference is None or getattr(inference, "empty", True):
            return jsonify({"success": False, "error": "Inference DataFrame missing/empty."}), 400

        try:
            final_df = gc.one_hot_encode_columns(
                imputed,
                inference,
                df_anomalies=anomaly,
                df_personal=personal,
                target_col=target,
            )
            readable_base_df = imputed.copy()
        except Exception as e:
            return jsonify({"success": False, "error": f"one_hot_encode_columns failed: {e}"}), 500

        if model_key == "rf":
            try:
                out = rftrain.train_random_forest(
                    df_encoded=final_df,
                    target_variable=target,
                    MODE=mode,
                    SCORING="accuracy",
                    N_SPLITS=5,
                    N_ITER=25,
                    use_cross_validation=use_cv,
                    time_limit_s=time_limit_s,
                    seed=seed,
                )
            except Exception as e:
                return jsonify({"success": False, "error": str(e)}), 500

            model_obj = out.pop("_model", None)
            if model_obj is None:
                model_obj = out.pop("model_object", None)

            split_indices = out.pop("split_indices", None)
            
            readable_split_data = None

            if split_indices:
                readable_split_data = {
                    "train_table": readable_base_df.loc[split_indices.get("train_idx", [])].copy(),
                    "validation_table": readable_base_df.loc[split_indices.get("validation_idx", [])].copy(),
                    "test_table": readable_base_df.loc[split_indices.get("test_idx", [])].copy(),
                }

            if model_obj is None:
                return jsonify({"success": False, "error": "RF training did not return model_object (needed for download)."}), 500

            filename = f"rf_{uuid.uuid4().hex}.zip"
            file_path = _artifact_path(filename)

            _build_model_zip_artifact(
                zip_path=file_path,
                model_obj=model_obj,
                result_obj=out,
                split_data=readable_split_data,
                model_filename="rf_model.pkl",
            )

            return jsonify({
                "success": True,
                "result": out,
                "artifact": {
                    "filename": filename,
                    "download_url": f"/model/download/{filename}",
                }
            })

        if model_key == "xgb":
            try:
                out = xgbtrain.train_xgboost(
                    df_encoded=final_df,
                    target_variable=target,
                    MODE=mode,
                    SCORING="accuracy",
                    N_SPLITS=5,
                    N_ITER=25,
                    use_cross_validation=use_cv,
                    time_limit_s=time_limit_s,
                    seed=seed,
                )
            except Exception as e:
                return jsonify({"success": False, "error": str(e)}), 500

            model_obj = out.pop("_model", None)
            if model_obj is None:
                model_obj = out.pop("model_object", None)

            split_indices = out.pop("split_indices", None)
            
            readable_split_data = None

            if split_indices:
                readable_split_data = {
                    "train_table": readable_base_df.loc[split_indices.get("train_idx", [])].copy(),
                    "validation_table": readable_base_df.loc[split_indices.get("validation_idx", [])].copy(),
                    "test_table": readable_base_df.loc[split_indices.get("test_idx", [])].copy(),
                }

            if model_obj is None:
                return jsonify({"success": False, "error": "XGB training did not return model_object (needed for download)."}), 500

            filename = f"xgb_{uuid.uuid4().hex}.zip"
            file_path = _artifact_path(filename)

            _build_model_zip_artifact(
                zip_path=file_path,
                model_obj=model_obj,
                result_obj=out,
                split_data=readable_split_data,
                model_filename="xgb_model.pkl",
            )

            return jsonify({
                "success": True,
                "result": out,
                "artifact": {
                    "filename": filename,
                    "download_url": f"/model/download/{filename}",
                }
            })


    preset = (req.get("automl_preset") or "best_quality").strip()

    hpo_mode = (req.get("search_mode") or "off").strip()
    if hpo_mode == "none":
        hpo_mode = "off"

    num_trials = int(req.get("num_trials") or 20)

    tmp_dir = tempfile.mkdtemp(prefix="agjob_")
    data_path = os.path.join(tmp_dir, "train.csv")
    out_json  = os.path.join(tmp_dir, "result.json")

    automl_filename = f"automl_{uuid.uuid4().hex}.zip"
    automl_artifact = _artifact_path(automl_filename)

    try:
        imputed.to_csv(data_path, index=False)

        cmd = [
            "/opt/ag-venv/bin/python",
            "/app/src/autogluon_train.py",
            "--data", data_path,
            "--label", target,
            "--preset", preset,
            "--hpo_mode", hpo_mode,
            "--time_limit_s", str(time_limit_s),
            "--seed", str(seed),
            "--num_trials", str(num_trials),
            "--out_json", out_json,
            "--artifact_path", automl_artifact,
        ]

        try:
            subprocess.run(
                cmd,
                check=True,
                capture_output=True,
                text=True,
            )
        except subprocess.CalledProcessError as e:
            return jsonify({
                "success": False,
                "error": "AutoML subprocess failed",
                "stdout": e.stdout,
                "stderr": e.stderr,
            }), 500

        if not os.path.isfile(out_json):
            return jsonify({"success": False, "error": "AutoML did not produce result.json"}), 500
        if not os.path.isfile(automl_artifact):
            return jsonify({"success": False, "error": "AutoML did not produce artifact zip"}), 500

        with open(out_json, "r", encoding="utf-8") as f:
            out = json.load(f)

        split_data = out.pop("split_data", None)

        if split_data:
            split_data = {
                "train_table": pd.DataFrame(split_data.get("train_table", [])),
                "validation_table": pd.DataFrame(split_data.get("validation_table", [])),
                "test_table": pd.DataFrame(split_data.get("test_table", [])),
            }

            _append_split_tables_to_existing_zip(automl_artifact, split_data)

        return jsonify({
            "success": True,
            "result": out,
            "artifact": {
                "filename": automl_filename,
                "download_url": f"/model/download/{automl_filename}",
            }
        })

    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)


@bp.route("/downloadZip", methods=["POST"])
def download_zip():
    req = request.get_json(silent=True) or {}
    selected = req.get("selected", [])
    zip_name = (req.get("zip_name") or "").strip()

    if not isinstance(selected, list):
        selected = []

    # Standardname für ZIP setzen
    if not zip_name:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        zip_name = f"metadata_bundle_{ts}.zip"

    #".zip" anhängen
    if not zip_name.lower().endswith(".zip"):
        zip_name = f"{zip_name}.zip"

    #allgemeine Infos aus der Session laden
    meta = session.get("data", {}) or {}
    filename = meta.get("name", "dataset")
    base_name = filename.split(".")[0] if isinstance(filename, str) else "dataset"
    target = meta.get("target_variable", "")

    #Zwischengespeicherte DataFrames aus der Session laden
    original = load_dataframe_from_session("original")
    imputed = load_dataframe_from_session("impute")
    inference = load_dataframe_from_session("inference")
    anomaly = load_dataframe_from_session("anomaly")
    personal_1 = load_dataframe_from_session("personal_1")
    

    #fehler bei keiner Auswahl im Downloadfenster
    if not selected:
        return jsonify({"success": False, "error": "No files selected."}), 400

    try:
        run_map = {}

        #Imputation-Metadaten -> imputierten DataFrame erzeugen
        if "imputation.json" in selected:
            if imputed is None:
                raise ValueError("Missing imputed dataframe")
            run_map["imputation.json"] = md.generate_metadata_json_imputation(imputed, base_name)

        #Feature-Type-Ergebnis JSON erzeugen
        if "feature_types.json" in selected:
            if inference is None:
                raise ValueError("Missing inference dataframe")

            feature_type_mapping = _feature_types_result_df_to_mapping(inference)
            run_map["feature_types.json"] = md.generate_metadata_json_feature_types(
                feature_type_mapping,
                base_name
            )

        #JSON erzeugen
        if "personal_data.json" in selected:
            if personal_1 is None:
                raise ValueError("Missing personal dataframe")

            personal_mapping = _personal_result_df_to_mapping(personal_1)
            run_map["personal_data.json"] = md.generate_metadata_json_personal_data(
                personal_mapping,
                base_name
            )

        #Anomaly-DataFrame zusammengefasst
        if "anomalies.json" in selected:
            if anomaly is None:
                raise ValueError("Missing anomaly dataframe")

            anomaly_summary = _anomaly_df_to_summary(anomaly)
            run_map["anomalies.json"] = md.generate_metadata_json_anomalies(
                anomaly_summary,
                base_name
            )

    except Exception as e:
        #Fehler Meldung falls Zip erzeugung fehlschlägt
        return jsonify({
            "success": False,
            "error": f"Missing one of the required DataFrames or JSON generation failed: {e}"
        }), 500

    cleaned_csv_bytes = None

    if "cleaned_data.csv" in selected:
        try:
            if "anomalies.json" in selected and "personal_data.json" not in selected:
                cleaned_df = gc.one_hot_encode_columns(
                    imputed, inference,
                    df_anomalies=anomaly,
                    target_col=target
                )

            elif "anomalies.json" not in selected and "personal_data.json" in selected:
                cleaned_df = gc.one_hot_encode_columns(
                    imputed, inference,
                    df_personal=personal_1,
                    target_col=target
                )

            elif "anomalies.json" in selected and "personal_data.json" in selected:
                cleaned_df = gc.one_hot_encode_columns(
                    imputed, inference,
                    df_anomalies=anomaly,
                    df_personal=personal_1,
                    target_col=target
                )

            else:
                cleaned_df = gc.one_hot_encode_columns(
                    imputed, inference,
                    target_col=target
                )

            #CSV vorbereiten um in ZIP geschrieben zu werden
            cleaned_csv_bytes = cleaned_df.to_csv(index=False).encode("utf-8")

        except Exception as e:
            return jsonify({"success": False, "error": f"cleaned_data.csv failed: {e}"}), 500

    #ZIP-Datei im Speicher erzeugen
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", compression=zipfile.ZIP_DEFLATED) as zf:

        #JSON-Dateien in ZIP File schreiben
        for fname in selected:
            if fname in run_map:
                obj = run_map[fname]
                payload = json.dumps(obj, indent=2, ensure_ascii=False).encode("utf-8")
                zf.writestr(f"metadata/{fname}", payload)

        if cleaned_csv_bytes is not None:
            zf.writestr("metadata/cleaned_data.csv", cleaned_csv_bytes)

    buf.seek(0)

    #ZIP zurückgeben
    return send_file(
        buf,
        as_attachment=True,
        download_name=zip_name,
        mimetype="application/zip"
    )



@bp.route("/reset", methods=["POST"])
def reset_session():
    # Delete only the current user's files.
    # This must run before session.clear(), because the workspace ID is
    # stored inside the Flask session.
    remove_current_workspace()

    session.clear()
    session.modified = True

    return jsonify({"success": True})
