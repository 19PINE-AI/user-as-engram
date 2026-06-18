# Data

All synthetic and self-contained (only LOCOMO is fetched separately). Everything
here is also regenerable — see [`../REPRODUCE.md`](../REPRODUCE.md) (Level 1).

| File / dir | Contents |
|---|---|
| `corpora.json` | The base 200-fact benchmark: `user_facts` (100) + `org_facts` (100) + `multi_users`. Each fact is `{trigger, prompt, gold, schema}`. |
| `corpora_xl.json` | 1,000 USER + 1,000 ORG templated facts (fact-count scaling sweeps). |
| `corpora_xxl.json` | 3,132 distinct trigger templates (high-density / distinct-template stress tests). |
| `users/uXXX.json` | 30 synthetic users (`u000`–`u029`). Each: `{uid, attrs, facts, direct_qa, indirect_qa}` — `facts` are atomic natural-language statements; `indirect_qa` items carry `required_fact_keys` (which facts the answer needs). `u000`–`u019` are test users; `u020`–`u029` are held out to train the shared LoRA. |
| `users_medical/mXXX.json` | 30 users in a third-person medical schema (the cross-schema generalization test). Same shape as `users/`. |
| `multihop_chains.json` | The chained-fact corpus for the multi-hop probe: `{description, items}`, each item `{facts: [[trigger, gold], …], query, expected, overlap}`. `overlap` marks whether the query ends in fact-2's trigger (a lucky word-match vs. a true chain). |
| `locomo10.json` | **Not shipped.** Third-party LOCOMO benchmark (Maharana et al., 2024, arXiv:2402.17753); obtain separately and place here. Only needed to *re-run* the LOCOMO evals — the LOCOMO figures already rebuild from `../results/`. |

All personas are fictional. Regenerate the per-user sets with
`code/lora_baseline/synth_users.py` / `synth_users_medical.py`, and the corpora
with `code/scripts/build_corpus*.py`.
