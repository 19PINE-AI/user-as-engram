import json, os
d = json.load(open(os.path.expanduser("~/clean-tests/results/T6/interim2.json")))["cells"]
ms = ["lora_down", "lora_down+reg", "lora_up", "lora_up+reg", "neuron", "neuron+reg", "neuron_bias", "neuron_bias+reg", "rome", "qkv", "qkv+reg"]
print("method | layer | n | direct | para | twohop | bleed [CI] | fire10 | text_kl | neighbor_kl | resid_frac")
for m in ms:
    for L in ["L3", "L7", "L15", "L23", "Lall"]:
        c = d.get(f"{m}|{L}")
        if not c:
            continue
        g = lambda k: c[k][0]
        print(f"{m} | {L} | {c['n']} | {g('direct'):.3f} | {g('para'):.3f} | {g('twohop'):.3f} | {g('bleed'):.4f} [{c['bleed'][1]:.4f},{c['bleed'][2]:.4f}] | {g('fire10'):.4f} | {g('text_kl'):.2e} | {g('neighbor_kl'):.3f} | {g('resid_frac'):.3f}")
