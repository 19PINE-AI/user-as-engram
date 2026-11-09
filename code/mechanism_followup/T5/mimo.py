"""MiMo Pro client: round-robin over box2/box3 tunnels, max 2 concurrent per box, JSON replies."""
import os, re, json, asyncio, random
from openai import AsyncOpenAI

def _env():
    d = {}
    for line in open(os.path.expanduser("~/clean-tests/mimo.env")):
        if "=" in line:
            k, v = line.strip().split("=", 1); d[k] = v
    return d

E = _env()
MODEL = E.get("MIMO_MODEL", "mimo-v2.6-pro-mopd-local")
_urls = E["MIMO_URLS"].split(",")
_keys = [E["MIMO_KEY_2"], E["MIMO_KEY_3"]]
CLIENTS = [AsyncOpenAI(base_url=u, api_key=k, timeout=300) for u, k in zip(_urls, _keys)]
PER_BOX = int(os.environ.get("MIMO_PER_BOX", "2"))
SEMS = None
_rr = 0


def _sems():
    global SEMS
    if SEMS is None:
        SEMS = [asyncio.Semaphore(PER_BOX) for _ in CLIENTS]
    return SEMS


def parse_json(text):
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.S)
    m = re.search(r"```(?:json)?\s*(.*?)```", text, flags=re.S)
    if m:
        text = m.group(1)
    s = text.find("{"); s2 = text.find("[")
    if s == -1 or (s2 != -1 and s2 < s):
        s = s2
    e = max(text.rfind("}"), text.rfind("]"))
    return json.loads(text[s:e + 1])


async def ask_json(prompt, max_tokens=3000, retries=4):
    global _rr
    sems = _sems()
    last = None
    for attempt in range(retries):
        i = _rr % len(CLIENTS); _rr += 1
        async with sems[i]:
            try:
                r = await CLIENTS[i].chat.completions.create(
                    model=MODEL, messages=[{"role": "user", "content": prompt}],
                    max_tokens=max_tokens, temperature=0.0)
                txt = r.choices[0].message.content or ""
                return parse_json(txt), txt
            except Exception as e:  # network or parse
                last = e
                await asyncio.sleep(2 + 3 * attempt + random.random())
    raise RuntimeError(f"mimo failed: {last!r}")
