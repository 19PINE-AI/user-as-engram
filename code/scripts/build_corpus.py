"""
P8 — Generate large synthetic fact corpora for User-as-Engram evaluation.

Produces three artifacts:
  1) 100 USER facts (per-user personal info, fictional entities)
  2) 100 ORG facts (per-organisation info, fictional entities)
  3) 100 simulated users, each with 30 facts (for multi-user leakage test)

Each fact: (trigger_phrase, prompt, gold_first_token_text)
Saved to $USER_AS_ENGRAM_ROOT/data/corpora.json
"""
import os
UAE_ROOT = os.environ.get("USER_AS_ENGRAM_ROOT") or (
    os.path.dirname(os.environ["NANOCHAT_BASE_DIR"]) if os.environ.get("NANOCHAT_BASE_DIR")
    else os.getcwd())
import json
import random
from pathlib import Path

random.seed(31337)


def gen_user_facts(n=100):
    NAMES   = ["Alex", "Sam", "Jordan", "Riley", "Morgan", "Quinn", "Avery", "Skyler", "Dakota", "Phoenix",
               "Reese", "Hayden", "Parker", "Drew", "Sloan", "Tate", "Logan", "Marlowe", "Onyx", "Pax"]
    SURNAMES = ["Patel", "Kim", "Garcia", "Diaz", "Singh", "Lopez", "Brown", "Lee", "Wilson", "Moore",
                "Taylor", "Anderson", "Walker", "Hall", "Young", "Wright", "Hill", "Scott"]
    SPICES  = ["saffron", "cardamom", "turmeric", "cumin", "fennel", "clove", "nutmeg", "paprika"]
    PETS    = ["cat", "dog", "rabbit", "ferret", "tortoise", "parrot", "hamster"]
    COLORS  = ["charcoal", "amber", "crimson", "emerald", "ivory", "jade", "russet", "ochre"]
    CITIES  = ["Portland", "Boston", "Austin", "Denver", "Boise", "Madison", "Tampa", "Tucson"]
    DAYS    = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
    BRANDS  = ["Subaru", "Toyota", "Honda", "Volvo", "Mazda", "Hyundai", "Kia"]
    SCHEMA = [
        ("My doctor's name is Dr.", " {S}",        SURNAMES, "doctor"),
        ("My dentist's name is Dr.", " {S}",       SURNAMES, "dentist"),
        ("My favorite spice is",    " {S}",        SPICES,   "spice"),
        ("My pet is a",              " {S}",        PETS,     "pet"),
        ("My favorite color is",    " {S}",        COLORS,   "color"),
        ("I live in",               " {S}",        CITIES,   "city"),
        ("My gym day is",           " {S}",        DAYS,     "gym_day"),
        ("My yoga day is",          " {S}",        DAYS,     "yoga_day"),
        ("I drive a",               " {S}",        BRANDS,   "car_brand"),
        ("My emergency contact is my sibling", " {S}", NAMES, "sibling"),
    ]
    facts = []
    for i in range(n):
        rng = random.Random(i * 991 + 7)
        trig_template, gold_template, vocab, schema = rng.choice(SCHEMA)
        gold = rng.choice(vocab)
        trig = trig_template
        prompt = trig
        gold_text = gold_template.format(S=gold)
        facts.append({"trigger": trig, "prompt": prompt, "gold": gold_text, "schema": schema})
    return facts


def gen_org_facts(n=100):
    ORGS = ["Globex", "Initech", "Hooli", "Stark", "Wayne", "Acme", "Vandelay", "Soylent",
            "Pied Piper", "Massive Dynamic", "Cyberdyne", "Tyrell", "Weyland", "Yoyodyne", "Aperture",
            "Dunder", "Krusty", "Rekall", "Buy n Large", "Initrode"]
    HOURS = ["7", "8", "9", "10", "11"]
    EXTS = ["1100", "2200", "3300", "4400", "5500", "6600", "7700", "8800", "9900"]
    ANIMALS = ["otter", "owl", "fox", "bear", "wolf", "raven", "lynx", "elk"]
    CITIES = ["Manhattan", "Boston", "Seattle", "Atlanta", "Denver", "Portland", "Chicago"]
    NAMES = ["Bruce", "Tony", "Diana", "Clark", "Peter", "Steve", "Natasha", "Wanda"]
    MONTHS = ["January", "February", "March", "April", "May", "June", "July", "August", "September", "October", "November", "December"]
    DEPTS = ["support", "sales", "billing", "legal", "engineering", "finance", "operations"]
    DAYS = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday"]
    SCHEMA = [
        ("{O} office hours start at",                          " {V}", HOURS,   "office_open"),
        ("{O} IT support extension is",                       " {V}", EXTS,    "it_ext"),
        ("{O}'s mascot animal is the",                         " {V}", ANIMALS, "mascot"),
        ("{O} headquarters is in",                             " {V}", CITIES,  "hq"),
        ("{O} CEO emeritus is",                                " {V}", NAMES,   "ceo_emeritus"),
        ("{O}'s fiscal year starts in",                        " {V}", MONTHS,  "fy_start"),
        ("{O}'s customer support email starts with",           " {V}", DEPTS,   "support_email"),
        ("{O}'s monthly all-hands is on the first",            " {V}", DAYS,    "allhands_day"),
        ("{O}'s flagship product is called",                   " {V}", ANIMALS, "flagship"),  # whimsical
        ("{O}'s parent company is",                            " {V}", ORGS,    "parent"),
    ]
    facts = []
    for i in range(n):
        rng = random.Random(i * 1009 + 3)
        org = rng.choice(ORGS)
        trig_template, gold_template, vocab, schema = rng.choice(SCHEMA)
        gold = rng.choice(vocab)
        trig = trig_template.format(O=org)
        prompt = trig
        gold_text = gold_template.format(V=gold)
        facts.append({"trigger": trig, "prompt": prompt, "gold": gold_text, "org": org, "schema": schema})
    return facts


def gen_multi_users(n_users=100, facts_per_user=30):
    NAMES = ["Alex", "Sam", "Jordan", "Riley", "Morgan", "Quinn", "Avery", "Skyler"]
    SURNAMES = ["Patel", "Kim", "Garcia", "Singh", "Lopez", "Brown", "Lee", "Wilson"]
    CITIES = ["Portland", "Boston", "Austin", "Denver", "Boise", "Madison"]
    SPICES = ["saffron", "cardamom", "turmeric", "cumin", "fennel"]
    COLORS = ["charcoal", "amber", "crimson", "emerald", "ivory", "jade"]
    DAYS = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday"]
    PETS = ["cat", "dog", "rabbit", "ferret"]
    BRANDS = ["Subaru", "Toyota", "Honda", "Volvo"]

    SCHEMA = [
        ("My doctor's name is Dr.", "{S}", SURNAMES, "doctor"),
        ("My dentist's name is Dr.", "{S}", SURNAMES, "dentist"),
        ("My favorite spice is", "{S}", SPICES, "spice"),
        ("I live in", "{S}", CITIES, "city"),
        ("My gym day is", "{S}", DAYS, "gym_day"),
        ("My yoga day is", "{S}", DAYS, "yoga_day"),
        ("My pet is a", "{S}", PETS, "pet_kind"),
        ("My favorite color is", "{S}", COLORS, "color"),
        ("My spouse is", "{S}", NAMES, "spouse"),
        ("My sibling is", "{S}", NAMES, "sibling"),
    ]
    users = []
    for u_id in range(n_users):
        rng = random.Random(u_id * 7919 + 17)
        # Each user has facts_per_user facts; we cycle through schemas with replacement
        facts = []
        for i in range(facts_per_user):
            trig_template, gold_template, vocab, schema = rng.choice(SCHEMA)
            gold = rng.choice(vocab)
            facts.append({
                "trigger": trig_template,
                "prompt": trig_template,
                "gold": " " + gold,
                "schema": schema,
            })
        users.append({"user_id": u_id, "facts": facts})
    return users


def main():
    out_dir = Path(f"{UAE_ROOT}/data")
    out_dir.mkdir(parents=True, exist_ok=True)
    user_facts = gen_user_facts(100)
    org_facts = gen_org_facts(100)
    multi_users = gen_multi_users(100, 30)
    bundle = {
        "user_facts": user_facts,
        "org_facts": org_facts,
        "multi_users": multi_users,
    }
    out_path = out_dir / "corpora.json"
    with open(out_path, "w") as f:
        json.dump(bundle, f, indent=2)
    print(f"Wrote {len(user_facts)} user facts, {len(org_facts)} org facts, "
          f"{len(multi_users)} users × {len(multi_users[0]['facts'])} facts to {out_path}")
    # Print a few samples
    print("\nSample USER facts:")
    for f in user_facts[:3]:
        print(f"  {f['trigger']!r:60s} -> {f['gold']!r}")
    print("\nSample ORG facts:")
    for f in org_facts[:3]:
        print(f"  {f['trigger']!r:60s} -> {f['gold']!r}")
    print(f"\nMulti-user 0 first 3 facts:")
    for f in multi_users[0]['facts'][:3]:
        print(f"  {f['trigger']!r:60s} -> {f['gold']!r}")


if __name__ == "__main__":
    main()
