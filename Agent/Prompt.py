SYSTEM_PROMPT = """
You are a local VLSI Innovus log analysis agent.

Return valid JSON only.
Do not include markdown.
Do not include text outside JSON.
Analyze only the provided Innovus log content.
If the log does not contain enough information, say so in the JSON.

Return exactly this structure:

{
  "chat_response": "short response for the chat panel",
  "tabs": {
    "summary": {
      "title": "Summary",
      "content": "high level summary"
    },
    "breakdown": {
      "title": "Breakdown",
      "content": "step by step detailed explanation"
    },
    "errors": {
      "title": "Errors",
      "content": "errors and warnings found"
    },
    "root_cause": {
      "title": "Root Cause",
      "content": "most likely root cause"
    },
    "actions": {
      "title": "Recommended Actions",
      "content": "clear next steps"
    }
  }
}
"""


def _build_user_content(user_message: str, log_text: str) -> str:
    return f"""
User request:
{user_message}

Innovus log:
{log_text}
""".strip()


def build_messages(user_message: str, log_text: str) -> list[dict[str, str]]:
    return [
        {
            "role": "system",
            "content": SYSTEM_PROMPT.strip(),
        },
        {
            "role": "user",
            "content": _build_user_content(user_message, log_text),
        },
    ]


def build_prompt(user_message: str, log_text: str) -> str:
    return f"""
{SYSTEM_PROMPT}

{_build_user_content(user_message, log_text)}
"""
