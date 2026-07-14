import base64
import json
import time
import requests
 
OLLAMA_URL = "http://localhost:11434/api/chat"
 
# IMPORTANT: set this to whatever tag `ollama list` shows on your machine.
# "gemma4" pulls the default tag; if you pulled a specific size use that,
# e.g. "gemma4:12b".
MODEL_NAME = "gemma4"
 
SYSTEM_PROMPT = """You are a construction-site safety auditor reviewing a snapshot
flagged by an automated PPE (Personal Protective Equipment) detector.
 
The detector only looks at bounding boxes for "Person", "Hardhat" and "Safety
Vest" and has no understanding of context, distance, or activity. It WILL be
wrong sometimes. Your job is to look at the actual image and decide whether
this is a REAL safety violation or a FALSE ALARM, using common sense.
 
Treat it as a FALSE ALARM (not a violation) when, for example:
- The worker is clearly resting, sitting, or stationary at a safe distance
  from any machinery, vehicles, excavation, scaffolding, or moving equipment.
- The worker is drinking water, eating, or on a break, away from active
  hazards -- even if a helmet or vest is missing in that moment.
- The "missing" item is actually present but occluded, an unusual color, or
  the detector clearly boxed the wrong region / wrong person.
- The person is in a designated rest/break area (e.g. under a shade tent, at
  a table, far from cranes, trucks, or excavators).
 
Treat it as a REAL VIOLATION when:
- The worker is actively working, walking through, or standing inside an
  active work zone near moving machinery, vehicles, cranes, scaffolding,
  open trenches, or overhead loads, without the required PPE.
- The worker is operating or standing right next to running equipment
  without a helmet and/or vest.
- You genuinely cannot tell the context (e.g. bad crop, blur) AND the PPE
  really does look absent -- in that case do NOT silently dismiss it; mark
  it for human review instead.
 
Respond with ONLY a single JSON object. No markdown fences, no commentary
before or after it. Use exactly this schema:
 
{
  "violation_confirmed": true or false,
  "needs_human_review": true or false,
  "category": "no_helmet" | "no_vest" | "no_helmet_and_vest" | "false_alarm" | "detector_error",
  "confidence": 0.0 to 1.0,
  "reasoning": "one or two plain-English sentences explaining the call",
  "context": {
    "worker_activity": "e.g. resting / walking / operating machinery / drinking water / unknown",
    "distance_from_hazard": "safe" | "close" | "unknown"
  }
}
"""
 
 
def _encode_image(image_path: str) -> str:
    with open(image_path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")
 
 
def _parse_json_response(raw: str) -> dict:
    """Gemma sometimes wraps JSON in ```json fences despite instructions -- strip them."""
    text = raw.strip()
    if text.startswith("```"):
        text = text.strip("`")
        if text.lower().startswith("json"):
            text = text[4:]
    return json.loads(text.strip())
 
 
def verify_violation(image_path: str, detector_flag: dict, retries: int = 2, timeout: int = 60) -> dict:
    """
    Send a snapshot to the local Gemma 4 model and get back a structured verdict.
 
    Parameters
    ----------
    image_path : path to the snapshot (should include some surrounding
        context, not just a tight crop of the person, so Gemma can judge
        distance from hazards / activity).
    detector_flag : dict describing what the upstream detector flagged, e.g.
        {"no_helmet": True, "no_vest": False, "track_id": 7}
 
    Returns
    -------
    dict matching the schema described in SYSTEM_PROMPT. If Gemma cannot be
    reached or returns unparseable output after retries, fails SAFE: marks
    the event for human review rather than silently dropping it.
    """
    user_text = (
        "The automated detector flagged this worker for: "
        f"{json.dumps(detector_flag)}. "
        "Look at the full image and give your verdict as JSON per the schema."
    )
 
    payload = {
        "model": MODEL_NAME,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_text, "images": [_encode_image(image_path)]},
        ],
        "stream": False,
        "options": {"temperature": 0.1},
    }
 
    last_error = None
    for _ in range(retries + 1):
        try:
            resp = requests.post(OLLAMA_URL, json=payload, timeout=timeout)
            resp.raise_for_status()
            raw = resp.json()["message"]["content"]
            return _parse_json_response(raw)
        except (requests.RequestException, KeyError, json.JSONDecodeError) as e:
            last_error = e
            time.sleep(1)
 
    # Fail SAFE: if Gemma is unreachable / returns garbage, don't silently
    # dismiss the detector's original flag -- send it to human review.
    return {
        "violation_confirmed": bool(detector_flag.get("no_helmet") or detector_flag.get("no_vest")),
        "needs_human_review": True,
        "category": "llm_error",
        "confidence": 0.0,
        "reasoning": f"Gemma verification failed after {retries + 1} attempt(s): {last_error}",
        "context": {"worker_activity": "unknown", "distance_from_hazard": "unknown"},
    }
 
 
if __name__ == "__main__":
    # Quick manual test: python gemma_verifier.py path/to/snapshot.jpg
    import sys
    if len(sys.argv) < 2:
        print("Usage: python gemma_verifier.py <image_path>")
        sys.exit(1)
    result = verify_violation(sys.argv[1], {"no_helmet": True, "no_vest": False, "track_id": 0})
    print(json.dumps(result, indent=2))