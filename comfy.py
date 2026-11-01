"""Minimal ComfyUI API client: queue a graph (API format) and wait for its outputs."""
import json, time, urllib.error, urllib.request
import settings

COMFY, CIN, COUT = settings.COMFY_URL, settings.COMFY_INPUT, settings.COMFY_OUTPUT


def post(g):
    settings.need_comfy()
    r = urllib.request.Request(f"{COMFY}/prompt", data=json.dumps({"prompt": g}).encode(), headers={"Content-Type": "application/json"})
    try:
        return json.loads(urllib.request.urlopen(r).read())["prompt_id"]
    except urllib.error.HTTPError as e:
        raise SystemExit(e.read().decode()[:2000])


def wait(pid):
    """Block until the prompt finishes; returns (seconds, outputs by node id)."""
    t = time.time()
    while True:
        h = json.loads(urllib.request.urlopen(f"{COMFY}/history/{pid}").read())
        if pid in h:
            st = h[pid]["status"]
            if st["status_str"] != "success":
                raise SystemExit(json.dumps(st["messages"])[-2000:])
            return time.time() - t, h[pid]["outputs"]
        time.sleep(1)
