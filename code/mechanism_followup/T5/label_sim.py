"""Label neurons (essay-style, blind to model) and simulate activations from the label (Gao-style).
Input: <dir>/neurons.json from collect.py. Output: <dir>/labels.json, <dir>/sims.json
Resumable: skips neurons already present.
"""
import argparse, asyncio, json, os, math
import numpy as np
from mimo import ask_json

ap = argparse.ArgumentParser()
ap.add_argument("--dir", required=True)
ap.add_argument("--stage", choices=["label", "sim", "both"], default="both")
ap.add_argument("--limit", type=int, default=0)
args = ap.parse_args()

neurons = json.load(open(os.path.join(args.dir, "neurons.json")))
if args.limit:
    neurons = neurons[:args.limit]
LP = os.path.join(args.dir, "labels.json"); SP = os.path.join(args.dir, "sims.json")
labels = json.load(open(LP)) if os.path.exists(LP) else {}
sims = json.load(open(SP)) if os.path.exists(SP) else {}
key = lambda n: f"{n['layer']}:{n['neuron']}"


def ctx_line(w, val=None):
    toks = [t.replace("\n", "\\n") for t in w["toks"]]
    m = w["mark"]
    toks[m] = f"«{toks[m]}»"
    s = "".join(toks)
    return s if val is None else f"[{val}] {s}"


def label_prompt(n):
    mx = max(n["max_post"], 1e-6)
    top = "\n".join(ctx_line(w, round(10 * max(w["pre"], 0) ** 2 / mx, 1)) for w in n["top"])
    bot = "\n".join(ctx_line(w) for w in n["bottom"][:10])
    return f"""You are auditing one MLP neuron in a small language model (interpretability research).
The neuron's activation is relu(pre)^2. Below are contexts where it fires most strongly; the token where it fires is marked «like this», with activation scaled 0-10 in brackets.

TOP-ACTIVATING CONTEXTS:
{top}

NEGATIVE POLE (contexts where its pre-activation is most negative, i.e. what it is pushed away from; marked token):
{bot}

Output tokens its output direction most PROMOTES: {n['logit_up']}
Output tokens it most SUPPRESSES: {n['logit_down']}

Task: describe what this neuron does.
- "label": one concise description (<= 25 words) precise enough that someone could predict its activation on new text.
- "negative_pole": short description of the negative side (or "none apparent").
- "verdict": exactly one of "one_job" (a single coherent job, possibly broad), "two_related" (two jobs with a clear relation, e.g. a concept and a related offshoot), "mixed_unrelated" (several unrelated things), "unclear" (no describable pattern).
- "category": exactly one of "entity_fact" (specific named entities, people, places, dates, numbers, factual associations), "lexical_token" (specific tokens/subwords/spelling), "syntax_format" (grammar, punctuation, formatting, code structure), "semantic_topic" (topic, concept, domain), "other".
- "confidence": 0-1.
Reply with only a JSON object with those keys."""


def sim_prompt(n, lab):
    blocks = []
    for i, w in enumerate(n["sim_windows"]):
        toks = [t.replace("\n", "\\n") for t in w["toks"]]
        blocks.append(f"Text {i}: " + " | ".join(f"{j}:{t}" for j, t in enumerate(toks)))
    return f"""A neuron in a language model has this description: "{lab['label']}".
For each text below (tokens are numbered and separated by " | "), predict the neuron's activation on EVERY token, as an integer 0-10 (0 = not active, 10 = maximally active). Most tokens are usually 0. The model reads left to right, so a token's activation can depend on earlier tokens.

{chr(10).join(blocks)}

Reply with only a JSON object: {{"0": [one integer per token of Text 0], "1": [...], ...}} with exactly as many integers as tokens in each text."""


async def do_label(n):
    try:
        lab, _ = await ask_json(label_prompt(n), max_tokens=1500)
        labels[key(n)] = lab
    except Exception as e:
        labels[key(n)] = {"error": repr(e)}


async def do_sim(n):
    lab = labels.get(key(n))
    if not lab or "label" not in lab:
        return
    try:
        out, _ = await ask_json(sim_prompt(n, lab), max_tokens=4000)
        preds = []
        for i, w in enumerate(n["sim_windows"]):
            p = out.get(str(i), [])
            p = [float(x) for x in p][:len(w["toks"])]
            p += [0.0] * (len(w["toks"]) - len(p))
            preds.append(p)
        real = np.concatenate([np.array(w["acts"]) for w in n["sim_windows"]])
        sim = np.concatenate([np.array(p) for p in preds])
        corr = float(np.corrcoef(real, sim)[0, 1]) if real.std() > 0 and sim.std() > 0 else 0.0
        sims[key(n)] = {"preds": preds, "corr": corr}
    except Exception as e:
        sims[key(n)] = {"error": repr(e)}


async def run(fn, todo):
    done = 0
    async def wrap(n):
        nonlocal done
        await fn(n); done += 1
        if done % 20 == 0:
            json.dump(labels, open(LP, "w")); json.dump(sims, open(SP, "w"))
            print(f"{fn.__name__} {done}/{len(todo)}", flush=True)
    await asyncio.gather(*(wrap(n) for n in todo))
    json.dump(labels, open(LP, "w")); json.dump(sims, open(SP, "w"))


async def main():
    if args.stage in ("label", "both"):
        todo = [n for n in neurons if "label" not in labels.get(key(n), {})]
        await run(do_label, todo)
    if args.stage in ("sim", "both"):
        todo = [n for n in neurons if "corr" not in sims.get(key(n), {})]
        await run(do_sim, todo)

asyncio.run(main())
