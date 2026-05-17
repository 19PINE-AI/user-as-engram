"""
T1.4 — Cross-user hash collision audit.

Question: if N synthetic users each have ~30 facts, and we hash all the
facts to (layer, head, slot) addresses, how often does a specific
*user-distinguishing* token from user A land in the same slot as a
*user-distinguishing* token from user B?

This matters because per-user surgical insertion writes to specific
slots; if user A and user B's name/doctor/spouse facts share slots,
either (a) overwriting one corrupts the other, or (b) we need per-user
salt in the hash (a tractable fix but a real design constraint).

We distinguish two collision regimes:
  R1 — Total slot overlap. Includes shared common-token addresses
       (e.g., "my", "is", "I am", "born"). Expected: very high, mostly
       benign because we never want to put user-specific values in
       these.
  R2 — User-distinguishing-token overlap. We restrict to tokens that
       are unique per user (names, places, dates, attributes).
       Expected: low if hash is well-distributed.

Procedure:
  1. Programmatically build N=100 synthetic users using a small per-field
     generator (names, years, cities, etc.) — same template as T1.2 but
     with fresh values.
  2. For each user, render facts to text, hash, collect addresses
     touched, AND collect "key-token" addresses (the addresses arising
     from N-grams whose final token is the unique-value token, e.g.,
     "Patel" not "doctor").
  3. Compare cross-user collision rates for both R1 and R2.
"""
import sys, json, random
from pathlib import Path
from collections import defaultdict, Counter

sys.path.insert(0, str(Path(__file__).parent.parent / "refs"))
from engram_demo_v1 import NgramHashMapping, engram_cfg  # noqa

import numpy as np
from transformers import AutoTokenizer

random.seed(7)
np.random.seed(7)

# --- Synthetic user generators ---
FIRST = ["Alex","Sam","Jordan","Casey","Morgan","Riley","Kai","Rowan","Quinn","Sage",
        "Avery","Skyler","Dakota","Phoenix","Nova","Eden","Reese","Hayden","Parker","Emerson",
        "Drew","Lane","Sloan","Tate","Elliot","Finley","Harper","Indigo","Jace","Kendall",
        "Logan","Marlowe","Noor","Ocean","Presley","Quincy","Robin","Saylor","Tatum","Uri",
        "Vesper","Wren","Xeno","Yael","Zane","Bell","Cleo","Dakari","Echo","Fable",
        "Gable","Hollis","Iver","Juno","Kelby","Lior","Maven","Nico","Onyx","Pax",
        "Quill","Reign","Sable","Thorne","Umber","Vale","West","Xander","Yarrow","Zion",
        "Arden","Briar","Cypress","Dune","Ember","Fox","Gale","Haven","Iris","Jet",
        "Kit","Lark","Moss","North","Oak","Pine","Quill","Ridge","Slate","Teal",
        "Vega","Wolf","Yew","Zephyr","Aspen","Brook","Cedar","Dell","Elm","Fern"]
LAST  = ["Chen","Patel","Kim","Smith","Garcia","Diaz","Khan","Singh","Lopez","Martin",
        "Brown","Jones","Davis","Lee","Wilson","Moore","Taylor","Anderson","Thomas","Jackson",
        "White","Harris","Clark","Lewis","Walker","Hall","Allen","Young","King","Wright",
        "Hill","Scott","Green","Adams","Baker","Nelson","Carter","Perez","Roberts","Turner",
        "Phillips","Campbell","Parker","Evans","Edwards","Collins","Stewart","Sanchez","Morris","Rogers"]
CITY  = ["Portland","Seattle","Boston","Austin","Denver","Phoenix","Miami","Atlanta","Boise","Madison",
        "Tampa","Reno","Tulsa","Omaha","Tucson","Spokane","Eugene","Burlington","Olympia","Helena"]
COLOR = ["charcoal","amber","crimson","emerald","ivory","sapphire","jade","russet","ochre","cobalt"]
PET   = ["cat","dog","rabbit","parrot","tortoise","ferret","gecko"]
JOB   = ["software engineer","teacher","nurse","architect","accountant","barista","journalist","chef"]
COMPANY = ["Globex","Initech","Hooli","Stark","Wayne","Acme","Umbrella","Vandelay","Dunder","Soylent"]

ALLERGIES = ["peanuts","shellfish","eggs","dairy","gluten","sesame","soy"]
DAYS = ["Monday","Tuesday","Wednesday","Thursday","Friday","Saturday","Sunday"]
ACTIVITIES = ["gym","yoga","running","cycling","swimming","climbing","tennis"]

def gen_user(uid: int):
    rng = random.Random(uid * 991 + 17)
    first = rng.choice(FIRST)
    last  = rng.choice(LAST)
    spouse_first = rng.choice(FIRST)
    child_first  = rng.choice(FIRST)
    pet_name     = rng.choice(FIRST)
    doctor_last  = rng.choice(LAST)
    dentist_last = rng.choice(LAST)
    sister_first = rng.choice(FIRST)
    city = rng.choice(CITY)
    company = rng.choice(COMPANY)
    job = rng.choice(JOB)
    color = rng.choice(COLOR)
    pet  = rng.choice(PET)
    allergy1 = rng.choice(ALLERGIES); allergy2 = rng.choice([a for a in ALLERGIES if a!=allergy1])
    gym_day = rng.choice(DAYS); yoga_day = rng.choice([d for d in DAYS if d!=gym_day])
    activity = rng.choice(ACTIVITIES)
    birth = rng.randint(1970, 2000)
    spouse_birth = rng.randint(1970, 2000)
    child_birth  = rng.randint(2010, 2024)
    pet_birth    = rng.randint(2015, 2024)

    facts = [
        f"My name is {first} {last}.",
        f"I was born in {birth}.",
        f"I live in {city}.",
        f"My spouse is {spouse_first} {last}.",
        f"{spouse_first} was born in {spouse_birth}.",
        f"We have one child named {child_first}.",
        f"{child_first} was born in {child_birth}.",
        f"My doctor is Dr {doctor_last}.",
        f"My dentist is Dr {dentist_last}.",
        f"I am allergic to {allergy1}.",
        f"I am allergic to {allergy2}.",
        f"My emergency contact is my sister {sister_first}.",
        f"I work as a {job}.",
        f"I work at {company} Corp.",
        f"My gym day is {gym_day}.",
        f"My {activity} day is {yoga_day}.",
        f"My favorite color is {color}.",
        f"My pet is a {pet} named {pet_name}.",
        f"{pet_name} was born in {pet_birth}.",
    ]

    distinguishing_tokens = [first, last, spouse_first, child_first, pet_name,
                             doctor_last, dentist_last, sister_first, city,
                             company, str(birth), str(spouse_birth),
                             str(child_birth), str(pet_birth)]
    return {"uid": uid, "facts": facts, "key_tokens": distinguishing_tokens}

def main():
    N_USERS = 100
    hm = NgramHashMapping(
        engram_vocab_size=engram_cfg.engram_vocab_size,
        max_ngram_size=engram_cfg.max_ngram_size,
        n_embed_per_ngram=engram_cfg.n_embed_per_ngram,
        n_head_per_ngram=engram_cfg.n_head_per_ngram,
        layer_ids=engram_cfg.layer_ids,
        tokenizer_name_or_path=engram_cfg.tokenizer_name_or_path,
        pad_id=engram_cfg.pad_id,
        seed=engram_cfg.seed,
    )
    tokenizer = AutoTokenizer.from_pretrained(engram_cfg.tokenizer_name_or_path, trust_remote_code=True)

    total_slots = 0
    for layer_id in engram_cfg.layer_ids:
        for ngram_heads in hm.vocab_size_across_layers[layer_id]:
            total_slots += sum(ngram_heads)

    # For each user, collect:
    #   r1_user[uid] = set of all addresses touched by any fact
    #   r2_user[uid] = set of addresses where the suffix N-gram ENDS at a key-token
    r1_user = {}
    r2_user = {}

    print(f"Hashing {N_USERS} synthetic users (~30 facts each)...")
    for uid in range(N_USERS):
        u = gen_user(uid)
        all_addrs = set()
        key_addrs = set()
        # Tokenize key tokens once to detect them downstream
        # (We compare token-id sequences for membership.)
        key_token_ids_sets = []
        for kt in u["key_tokens"]:
            # Most tokens don't have a leading space; with leading space they tokenize differently
            for variant in (kt, " " + kt):
                ids = tokenizer.encode(variant, add_special_tokens=False)
                if ids:
                    key_token_ids_sets.append(tuple(ids))
        # Hash each fact
        for fact in u["facts"]:
            ids = tokenizer(fact, return_tensors="np").input_ids  # [1, T]
            T = ids.shape[1]
            hashes_per_layer = hm.hash(ids)
            # For "key" addresses, we need to know which positions are at the END of a key token sequence
            id_seq = ids[0].tolist()
            key_end_positions = set()
            for kt_ids in key_token_ids_sets:
                klen = len(kt_ids)
                if klen == 0: continue
                for i in range(T - klen + 1):
                    if tuple(id_seq[i:i+klen]) == kt_ids:
                        key_end_positions.add(i + klen - 1)
            for layer_id, h_arr in hashes_per_layer.items():
                B, Tt, H = h_arr.shape
                for t in range(Tt):
                    for h_idx in range(H):
                        addr = (layer_id, h_idx, int(h_arr[0, t, h_idx]))
                        all_addrs.add(addr)
                        if t in key_end_positions:
                            key_addrs.add(addr)
        r1_user[uid] = all_addrs
        r2_user[uid] = key_addrs

    # --- R1 collision: count shared addresses across users
    addr_user_count_r1 = Counter()
    for uid, addrs in r1_user.items():
        for addr in addrs:
            addr_user_count_r1[addr] += 1
    addr_user_count_r2 = Counter()
    for uid, addrs in r2_user.items():
        for addr in addrs:
            addr_user_count_r2[addr] += 1

    n_r1_total = len(addr_user_count_r1)
    n_r1_shared = sum(1 for _, n in addr_user_count_r1.items() if n > 1)
    n_r1_universal = sum(1 for _, n in addr_user_count_r1.items() if n == N_USERS)

    n_r2_total = len(addr_user_count_r2)
    n_r2_shared = sum(1 for _, n in addr_user_count_r2.items() if n > 1)
    n_r2_universal = sum(1 for _, n in addr_user_count_r2.items() if n == N_USERS)

    # Mean pairwise overlap across users
    pairwise_r1 = []
    pairwise_r2 = []
    uids = list(r1_user.keys())
    for i in range(len(uids)):
        for j in range(i+1, len(uids)):
            a, b = uids[i], uids[j]
            if len(r1_user[a]):
                pairwise_r1.append(len(r1_user[a] & r1_user[b]) / len(r1_user[a]))
            if len(r2_user[a]):
                pairwise_r2.append(len(r2_user[a] & r2_user[b]) / max(len(r2_user[a]), 1))

    mean_pw_r1 = sum(pairwise_r1)/len(pairwise_r1) if pairwise_r1 else 0
    mean_pw_r2 = sum(pairwise_r2)/len(pairwise_r2) if pairwise_r2 else 0

    print("\n--- R1 (all addresses) ---")
    print(f"Distinct addresses across all users: {n_r1_total:,}")
    print(f"Shared by ≥2 users:                  {n_r1_shared:,}  ({n_r1_shared/n_r1_total:.2%})")
    print(f"Touched by all {N_USERS} users:           {n_r1_universal:,}")
    print(f"Mean pairwise overlap (|A∩B|/|A|):   {mean_pw_r1:.4f}")

    print("\n--- R2 (key-token addresses only — these matter most) ---")
    print(f"Distinct addresses across all users: {n_r2_total:,}")
    print(f"Shared by ≥2 users:                  {n_r2_shared:,}  ({n_r2_shared/max(n_r2_total,1):.2%})")
    print(f"Touched by all {N_USERS} users:           {n_r2_universal:,}")
    print(f"Mean pairwise overlap (|A∩B|/|A|):   {mean_pw_r2:.4f}")

    # Birthday-paradox-style expectation for R2:
    # If R2 addresses are uniform over P slots, and each user touches ~k of them,
    # P(collision between two users) ≈ k^2 / P (small-k approx). With k~|R2| per user,
    # P  effectively the slot-space size for relevant tables.

    out = {
        "config": {"n_users": N_USERS, "total_slots": total_slots},
        "R1_all_addresses": {
            "distinct": n_r1_total,
            "shared_by_two_or_more": n_r1_shared,
            "universal": n_r1_universal,
            "mean_pairwise_overlap": mean_pw_r1,
        },
        "R2_key_token_addresses": {
            "distinct": n_r2_total,
            "shared_by_two_or_more": n_r2_shared,
            "universal": n_r2_universal,
            "mean_pairwise_overlap": mean_pw_r2,
        },
    }
    out_path = Path(__file__).parent.parent / "results" / "t1_4_collisions.json"
    out_path.write_text(json.dumps(out, indent=2))
    print(f"\nSaved: {out_path}")

if __name__ == "__main__":
    main()
