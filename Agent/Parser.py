import json
import re


TAB_TITLES = {
    "summary": "Summary",
    "breakdown": "Breakdown",
    "errors": "Errors",
    "root_cause": "Root Cause",
    "actions": "Recommended Actions",
}


def build_agent_response(
    chat_response: str,
    *,
    summary: str,
    breakdown: str,
    errors: str,
    root_cause: str,
    actions: str,
) -> dict:
    return {
        "chat_response": chat_response,
        "tabs": {
            "summary": {
                "title": TAB_TITLES["summary"],
                "content": summary,
            },
            "breakdown": {
                "title": TAB_TITLES["breakdown"],
                "content": breakdown,
            },
            "errors": {
                "title": TAB_TITLES["errors"],
                "content": errors,
            },
            "root_cause": {
                "title": TAB_TITLES["root_cause"],
                "content": root_cause,
            },
            "actions": {
                "title": TAB_TITLES["actions"],
                "content": actions,
            },
        },
    }


def build_error_response(
    chat_response: str,
    *,
    summary: str,
    breakdown: str,
    errors: str,
    root_cause: str,
    actions: str,
) -> dict:
    return build_agent_response(
        chat_response,
        summary=summary,
        breakdown=breakdown,
        errors=errors,
        root_cause=root_cause,
        actions=actions,
    )


def _load_json_payload(raw_text: str) -> dict | None:
    text = raw_text.strip()

    if not text:
        return None

    candidates = [text]

    fenced_match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
    if fenced_match:
        candidates.append(fenced_match.group(1))

    first_brace = text.find("{")
    last_brace = text.rfind("}")

    if first_brace != -1 and last_brace != -1 and last_brace > first_brace:
        candidates.append(text[first_brace:last_brace + 1])

    for candidate in candidates:
        try:
            payload = json.loads(candidate)
        except json.JSONDecodeError:
            continue

        if isinstance(payload, dict):
            return payload

    return None


def _normalize_tab(tab_key: str, tab_value) -> dict:
    if isinstance(tab_value, dict):
        title = str(tab_value.get("title") or TAB_TITLES[tab_key])
        content = str(tab_value.get("content", ""))
    elif tab_value is None:
        title = TAB_TITLES[tab_key]
        content = ""
    else:
        title = TAB_TITLES[tab_key]
        content = str(tab_value)

    return {
        "title": title,
        "content": content,
    }


def parse_agent_response(raw_text: str) -> dict:
    payload = _load_json_payload(raw_text)

    if payload is None:
        return build_error_response(
            "The model returned an unstructured response.",
            summary=raw_text,
            breakdown="Could not parse model output as JSON.",
            errors="JSON parsing failed.",
            root_cause="Could not extract root cause because JSON parsing failed.",
            actions="Improve the prompt or use a model/backend that follows the JSON format reliably.",
        )

    tabs = payload.get("tabs") if isinstance(payload.get("tabs"), dict) else {}

    return {
        "chat_response": str(payload.get("chat_response") or "Analysis complete."),
        "tabs": {
            tab_key: _normalize_tab(tab_key, tabs.get(tab_key))
            for tab_key in TAB_TITLES
        },
    }
