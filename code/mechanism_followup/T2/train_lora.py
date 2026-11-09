"""T2 Part A step 1: per-user LoRA on Qwen3-8B (POLAR recipe from lora_baseline/stage_a.py),
two variants: qkv (repo default) and mlp (gate/up/down). Writes adapter + eval json."""
import os, sys, json, time, random, argparse
import torch
torch.cuda.set_per_process_memory_fraction(28/96)
from pathlib import Path
sys.path.insert(0, os.path.expanduser("~/clean-tests/user-as-engram/code/lora_baseline"))
from stage_a import UserNTPDataset, collate_fn, evaluate
from synth_users import load_user
from torch.utils.data import DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig, get_peft_model

ap = argparse.ArgumentParser()
ap.add_argument("--variants", default="qkv,mlp")
ap.add_argument("--uids", default=",".join(f"u{i:03d}" for i in range(20)))
ap.add_argument("--model", default="Qwen/Qwen3-8B")
ap.add_argument("--rank", type=int, default=64); ap.add_argument("--alpha", type=int, default=128)
ap.add_argument("--lr", type=float, default=1e-4); ap.add_argument("--epochs", type=int, default=15)
args = ap.parse_args()
R = Path(os.path.expanduser("~/clean-tests/results/T2"))
UD = Path(os.path.expanduser("~/clean-tests/user-as-engram/data/users"))
TM = {"qkv": ["q_proj", "k_proj", "v_proj"], "mlp": ["gate_proj", "up_proj", "down_proj"]}
tok = AutoTokenizer.from_pretrained(args.model)
if tok.pad_token is None: tok.pad_token = tok.eos_token
base = AutoModelForCausalLM.from_pretrained(args.model, dtype=torch.bfloat16, device_map="cuda")
base.gradient_checkpointing_enable(); base.enable_input_require_grads()
import itertools
for uid, variant in itertools.product(args.uids.split(","), args.variants.split(",")):
    tm = TM[variant]
    outd = R / "adapters" / variant / uid
    if (outd / "metrics.json").exists(): print("skip", uid); continue
    random.seed(42); torch.manual_seed(42)
    user = load_user(UD / f"{uid}.json")
    model = get_peft_model(base, LoraConfig(r=args.rank, lora_alpha=args.alpha, target_modules=tm,
                           lora_dropout=0.05, bias="none", task_type="CAUSAL_LM"))
    pre_path = R / "base_eval" / f"{uid}.json"
    if not pre_path.exists():
        pre_path.parent.mkdir(parents=True, exist_ok=True)
        with model.disable_adapter():
            json.dump(evaluate(model, tok, user, "cuda"), open(pre_path, "w"), indent=1)
    trainable = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(trainable, lr=args.lr, weight_decay=0.01)
    t0 = time.time()
    for ep in range(args.epochs):
        model.train()
        ds = UserNTPDataset(user, tok, num_samples=200, max_len=192, qa_ratio=0.5, seed=42 + ep)
        dl = DataLoader(ds, batch_size=4, shuffle=True, collate_fn=lambda b: collate_fn(b, tok, 192))
        tl = 0
        for b in dl:
            b = {k: v.cuda() for k, v in b.items()}
            loss = model(**b).loss; loss.backward()
            torch.nn.utils.clip_grad_norm_(trainable, 1.0); opt.step(); opt.zero_grad(); tl += loss.item()
        print(f"[{variant} {uid}] ep{ep+1} loss={tl/len(dl):.4f}", flush=True)
    model.eval()
    ev = evaluate(model, tok, user, "cuda")
    outd.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(str(outd))
    ev.update(uid=uid, variant=variant, train_s=time.time() - t0, rank=args.rank, alpha=args.alpha)
    json.dump(ev, open(outd / "metrics.json", "w"), indent=1)
    print(f"[{variant} {uid}] direct={ev['direct_acc']:.3f} indirect={ev['indirect_acc']:.3f} t={ev['train_s']:.0f}s", flush=True)
    base = model.unload()  # strip LoRA modules, restore plain base
    del model, opt; torch.cuda.empty_cache()
