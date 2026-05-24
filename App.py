from pathlib import Path

from flask import Flask, request, jsonify, send_from_directory

from Agent.LLM import LLMBackendError, create_llm
from Agent.Parser import build_error_response, parse_agent_response
from Agent.Prompt import build_messages


app = Flask(__name__, static_folder="Frontend")

llm = create_llm()
BASE_DIR = Path(__file__).resolve().parent
LOG_PATH = BASE_DIR / "test.log"


def _load_log_text() -> str:
    return LOG_PATH.read_text(encoding="utf-8")


@app.route("/")
def index():
    return send_from_directory("Frontend", "index.html")


@app.route("/chat", methods=["POST"])
def chat():
    data = request.get_json(silent=True) or {}
    user_message = str(data.get("message", "")).strip()

    if not user_message:
        return jsonify(
            build_error_response(
                "Please enter a request to analyze.",
                summary="No user request was provided.",
                breakdown="The /chat endpoint received an empty message.",
                errors="Empty request body.",
                root_cause="The frontend sent a blank prompt.",
                actions="Enter a question or analysis request and try again.",
            )
        ), 400

    try:
        log_text = _load_log_text()
        messages = build_messages(user_message, log_text)
        response = llm.invoke(messages)
        result = parse_agent_response(response)
        return jsonify(result)
    except FileNotFoundError:
        return jsonify(
            build_error_response(
                "The log file could not be loaded.",
                summary=f"Expected log file: {LOG_PATH.name}",
                breakdown="The app is configured to analyze the local Innovus log before calling the model.",
                errors=f"Missing file: {LOG_PATH}",
                root_cause="The configured log file does not exist.",
                actions="Place the target Innovus log at test.log or update the configured path in App.py.",
            )
        ), 500
    except LLMBackendError as exc:
        return jsonify(
            build_error_response(
                "The model backend could not generate an analysis.",
                summary="Local GGUF model initialization or inference failed.",
                breakdown="The request reached the model layer, but the configured backend could not run successfully.",
                errors=str(exc),
                root_cause="The local model backend is unavailable or incompatible with the current environment.",
                actions="Set LOCAL_GGUF_MODEL_PATH if you want to force a specific GGUF file. If you still hit CPU instruction errors, use a compatible llama.cpp build for this CPU or switch to another local inference backend.",
            )
        ), 503
    except Exception as exc:
        return jsonify(
            build_error_response(
                "The app hit an unexpected error.",
                summary="Unexpected server-side failure during chat processing.",
                breakdown="The request could not be completed because an unhandled exception occurred.",
                errors=f"{type(exc).__name__}: {exc}",
                root_cause="Unexpected application error.",
                actions="Review the traceback in the server logs and retry after fixing the failing code path.",
            )
        ), 500


if __name__ == "__main__":
    app.run(debug=True, port=5000)
