"""Minimal ComfyUI API client: queue a graph (API format) and wait for its outputs. Every request has a timeout and is retried, so
a request that never gets an answer cannot block a run forever."""
import json, time, urllib.error, urllib.request
import settings

COMFY, CIN, COUT = settings.COMFY_URL, settings.COMFY_INPUT, settings.COMFY_OUTPUT


def _request(req, timeout=60, tries=5):
    for k in range(tries):
        try:
            return urllib.request.urlopen(req, timeout=timeout).read()
        except urllib.error.HTTPError:
            raise
        except (urllib.error.URLError, TimeoutError, ConnectionError, OSError) as e:
            if k == tries - 1:
                raise SystemExit(f"ComfyUI at {COMFY} did not answer ({e}); is it running?")
            time.sleep(2 + 3 * k)


def post(g):
    settings.need_comfy()
    r = urllib.request.Request(f"{COMFY}/prompt", data=json.dumps({"prompt": g}).encode(), headers={"Content-Type": "application/json"})
    try:
        return json.loads(_request(r))["prompt_id"]
    except urllib.error.HTTPError as e:
        raise SystemExit(e.read().decode()[:2000])


def wait(pid, limit=3 * 3600):
    """Block until the prompt finishes; returns (seconds, outputs by node id). Gives up after `limit` seconds, or when the prompt
    is neither queued nor in the history any more (ComfyUI restarted or dropped it)."""
    t = time.time(); missing = 0
    while time.time() - t < limit:
        h = json.loads(_request(f"{COMFY}/history/{pid}"))
        if pid in h:
            st = h[pid]["status"]
            if st["status_str"] != "success":
                raise SystemExit(json.dumps(st["messages"])[-2000:])
            return time.time() - t, h[pid]["outputs"]
        q = json.loads(_request(f"{COMFY}/queue"))
        queued = any(item[1] == pid for item in q.get("queue_running", []) + q.get("queue_pending", []))
        missing = 0 if queued else missing + 1
        if missing > 30:
            raise SystemExit(f"prompt {pid} is no longer queued in ComfyUI and has no result")
        time.sleep(1)
    raise SystemExit(f"prompt {pid} did not finish within {limit} s")
