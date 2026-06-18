"""
XXL fact corpus: 1000+ unique trigger templates for proper within-user
fact-density scaling.

The XL corpus capped at ~83 unique triggers because USER schemas were
all "My X is [V]" patterns with no name slots. Here we add name-bearing
templates (e.g., "My friend Alex's spice is X") so a single user can
plausibly have 1000 distinct fact triggers.

Total trigger pool target: ~6000 unique templates.

Saved to $USER_AS_ENGRAM_ROOT/data/corpora_xxl.json
"""
import os
UAE_ROOT = os.environ.get("USER_AS_ENGRAM_ROOT") or (
    os.path.dirname(os.environ["NANOCHAT_BASE_DIR"]) if os.environ.get("NANOCHAT_BASE_DIR")
    else os.getcwd())
import json
import random
from pathlib import Path

random.seed(20260506)


# Pools (subset of build_corpus_xl, slightly expanded)
NAMES = ["Alex","Sam","Jordan","Casey","Morgan","Riley","Kai","Rowan","Quinn","Sage",
         "Avery","Skyler","Dakota","Phoenix","Nova","Eden","Reese","Hayden","Parker","Emerson",
         "Drew","Lane","Sloan","Tate","Elliot","Finley","Harper","Indigo","Jace","Kendall",
         "Logan","Marlowe","Noor","Ocean","Presley","Quincy","Robin","Saylor","Tatum","Uri",
         "Vesper","Wren","Xeno","Yael","Zane","Bell","Cleo","Dakari","Echo","Fable",
         "Gable","Hollis","Iver","Juno","Kelby","Lior","Maven","Nico","Onyx","Pax",
         "Quill","Reign","Sable","Thorne","Umber","Vale","West","Xander","Yarrow","Zion"]
SURNAMES = ["Patel","Kim","Garcia","Diaz","Singh","Lopez","Brown","Lee","Wilson","Moore",
            "Taylor","Anderson","Thomas","Jackson","White","Harris","Clark","Lewis","Walker",
            "Hall","Allen","Young","King","Wright","Hill","Scott","Green","Adams","Baker",
            "Nelson","Carter","Perez","Roberts","Turner"]
SPICES = ["saffron","cardamom","turmeric","cumin","fennel","clove","nutmeg","paprika","ginger","cinnamon",
          "anise","sumac","oregano","basil","thyme","sage","mint","rosemary","tarragon","chive"]
COLORS = ["charcoal","amber","crimson","emerald","ivory","sapphire","jade","russet","ochre","cobalt",
          "scarlet","violet","teal","magenta","indigo","gold","silver","copper","bronze","mint"]
CITIES = ["Portland","Seattle","Boston","Austin","Denver","Boise","Madison","Tampa","Reno","Tulsa",
          "Omaha","Tucson","Spokane","Eugene","Burlington","Olympia","Helena","Nashville","Memphis","Houston"]
DAYS = ["Monday","Tuesday","Wednesday","Thursday","Friday","Saturday","Sunday"]
DRINKS = ["coffee","tea","matcha","kombucha","cider","ale","stout","lager","cabernet","chardonnay","cocoa","horchata"]
FRUITS = ["apple","banana","mango","papaya","kiwi","peach","plum","fig","cherry","grape","lychee","durian"]
ANIMALS = ["otter","owl","fox","bear","wolf","raven","lynx","elk","hawk","eagle","panda","tiger","lion","heron","stork"]
INSTRUMENTS = ["piano","guitar","violin","drums","saxophone","flute","cello","trumpet","harp","clarinet"]
SPORTS = ["tennis","basketball","soccer","baseball","football","golf","swimming","climbing","skiing","cycling"]
HOBBIES = ["chess","painting","running","cycling","cooking","gardening","photography","reading","hiking","yoga"]
JOBS = ["engineer","teacher","nurse","architect","accountant","barista","journalist","chef","dentist","lawyer","writer","designer"]
COMPANIES = ["Globex","Initech","Hooli","Stark","Wayne","Acme","Umbrella","Vandelay","Dunder","Soylent",
             "Pied Piper","Massive Dynamic","Cyberdyne","Tyrell","Weyland","Yoyodyne","Aperture","Buy n Large","Initrode","Krusty",
             "Rekall","Sirius Cybernetics","Nakatomi","Spacely","Cogswell","Lunar","Helios","Atlas","Apex","Nova Corp"]
MONTHS = ["January","February","March","April","May","June","July","August","September","October","November","December"]
LANGS = ["Spanish","French","German","Mandarin","Japanese","Korean","Italian","Portuguese","Russian","Arabic"]
COUNTRIES = ["Canada","France","Japan","Germany","Mexico","Italy","Spain","Brazil","India","Australia"]


# Name-bearing templates: each populates {N} with a name. With 70 names × ~30
# templates we get 2100 unique triggers from this category alone.
NAME_TEMPLATES = [
    ("My friend {N}'s favorite spice is",            SPICES,    "friend_spice"),
    ("My friend {N}'s favorite color is",            COLORS,    "friend_color"),
    ("My friend {N}'s favorite drink is",            DRINKS,    "friend_drink"),
    ("My friend {N}'s favorite fruit is",            FRUITS,    "friend_fruit"),
    ("My friend {N}'s favorite sport is",            SPORTS,    "friend_sport"),
    ("My friend {N}'s favorite instrument is the",   INSTRUMENTS,"friend_instrument"),
    ("My friend {N}'s favorite hobby is",            HOBBIES,   "friend_hobby"),
    ("My friend {N}'s favorite city is",             CITIES,    "friend_city"),
    ("My friend {N}'s favorite animal is the",       ANIMALS,   "friend_animal"),
    ("My cousin {N} lives in",                       CITIES,    "cousin_city"),
    ("My cousin {N} works as a",                     JOBS,      "cousin_job"),
    ("My uncle {N}'s favorite month is",             MONTHS,    "uncle_month"),
    ("My aunt {N}'s favorite language is",           LANGS,     "aunt_lang"),
    ("My nephew {N} plays the",                      INSTRUMENTS,"nephew_instrument"),
    ("My niece {N} studies in",                      COUNTRIES, "niece_country"),
    ("My neighbor {N} drives a",                     COMPANIES, "neighbor_car_brand"),  # using companies as placeholder
    ("My boss {N} prefers the cuisine of",           COUNTRIES, "boss_cuisine"),
    ("My team lead {N}'s standup day is",            DAYS,      "team_standup"),
    ("My mentor {N}'s book genre is",                ["fantasy","mystery","romance","thriller","biography","poetry"], "mentor_genre"),
    ("My therapist {N}'s clinic is in",              CITIES,    "therapist_city"),
    ("My doctor {N}'s specialty is",                 ["cardiology","oncology","neurology","dermatology","orthopedics","pediatrics"], "doctor_specialty"),
    ("My dentist {N}'s clinic is in",                CITIES,    "dentist_city"),
    ("My pediatrician {N} works at",                 COMPANIES, "ped_employer"),
    ("My optometrist {N}'s clinic is in",            CITIES,    "opt_city"),
    ("My yoga teacher {N}'s favorite asana is",      ["downward dog","tree pose","warrior","sun salutation","plank","cobra"], "yoga_asana"),
    ("My gym partner {N} prefers",                   SPORTS,    "gym_sport"),
    ("My pen pal {N} writes from",                   COUNTRIES, "penpal_country"),
    ("My godmother {N}'s favorite holiday is",       MONTHS,    "godmother_holiday"),
    ("My godfather {N}'s career is in",              JOBS,      "godfather_career"),
    ("My personal trainer {N}'s gym is in",          CITIES,    "trainer_gym"),
    ("My pet {N} is a",                              ["cat","dog","rabbit","ferret","tortoise","parrot","hamster","goldfish"], "pet_kind"),
    ("My favorite barista {N} works at",             COMPANIES, "barista_employer"),
    ("My doctoral advisor {N}'s field is",           ["physics","chemistry","biology","economics","statistics","linguistics"], "advisor_field"),
]

# Org-bearing templates with internal value selectors
ORG_TEMPLATES = [
    ("{C} office hours start at",                          ["7","8","9","10","11"], "office_open"),
    ("{C} IT support extension is",                        ["1100","2200","3300","4400","5500","6600","7700","8800","9900"], "it_ext"),
    ("{C}'s mascot animal is the",                         ANIMALS, "mascot"),
    ("{C} headquarters is in",                             CITIES,  "hq"),
    ("{C} CEO emeritus is",                                NAMES,   "ceo_emeritus"),
    ("{C}'s fiscal year starts in",                        MONTHS,  "fy_start"),
    ("{C}'s training day is",                              DAYS,    "training_day"),
    ("{C}'s data center is in",                            CITIES,  "datacenter"),
    ("{C}'s main color is",                                COLORS,  "main_color"),
    ("{C}'s annual retreat is in",                         CITIES,  "retreat_city"),
    ("{C}'s quarterly review is in",                       MONTHS,  "qr_month"),
    ("{C} primary language is",                            LANGS,   "primary_lang"),
    ("{C}'s annual conference takes place in",             CITIES,  "conf_city"),
    ("{C}'s code repository is named",                     ANIMALS, "repo_name"),
    ("{C}'s standup is every",                             DAYS,    "standup_day"),
    ("{C}'s remote-work day is",                           DAYS,    "remote_day"),
    ("{C}'s payroll provider is",                          COMPANIES,"payroll"),
    ("{C}'s preferred shipping carrier is",                COMPANIES,"shipper"),
    ("{C}'s sister company is",                            COMPANIES,"sister_co"),
    ("{C}'s legal counsel is",                             COMPANIES,"counsel"),
    ("{C}'s annual hackathon is in",                       MONTHS,  "hack_month"),
    ("{C}'s primary industry is",                          ["fintech","biotech","retail","education","logistics","gaming","media","energy"], "industry"),
    ("{C}'s flagship product mascot is the",               ANIMALS, "flagship_mascot"),
    ("{C}'s monthly all-hands is on the first",            DAYS,    "allhands_day"),
    ("{C}'s customer support email starts with",           ["support","sales","billing","legal","engineering","finance"], "support_email"),
    ("{C}'s breakfast is catered by",                      COMPANIES, "breakfast"),
    ("{C} parent company is",                              COMPANIES, "parent"),
]

# Plain (no slot) templates from original USER_SCHEMA
SIMPLE_TEMPLATES = [
    ("My favorite spice is",                                SPICES,     "spice"),
    ("My favorite color is",                                COLORS,     "color"),
    ("I live in",                                           CITIES,     "city"),
    ("My favorite drink is",                                DRINKS,     "drink"),
    ("My favorite fruit is",                                FRUITS,     "fruit"),
    ("My favorite sport is",                                SPORTS,     "sport"),
    ("My favorite hobby is",                                HOBBIES,    "hobby"),
    ("My instrument is the",                                INSTRUMENTS,"instrument"),
    ("My pet is a",                                         ["cat","dog","rabbit","ferret","tortoise","parrot"], "pet_kind"),
    ("I work as a",                                         JOBS,       "job"),
    ("My favorite season is",                               ["Spring","Summer","Fall","Winter"], "season"),
    ("My passport country is",                              COUNTRIES,  "passport"),
]


def gen_unique_triggers():
    """Enumerate all distinct (trigger_string, gold_value, schema_id) facts.
    Returns up to ~6000 unique fact templates."""
    facts = []
    # Name-bearing
    for tmpl, vocab, schema in NAME_TEMPLATES:
        for name in NAMES:
            trig = tmpl.format(N=name)
            for v in vocab:
                facts.append({"trigger": trig, "prompt": trig, "gold": " " + v, "schema": schema})
    # Org-bearing
    for tmpl, vocab, schema in ORG_TEMPLATES:
        for org in COMPANIES:
            trig = tmpl.format(C=org)
            for v in vocab:
                facts.append({"trigger": trig, "prompt": trig, "gold": " " + v, "schema": schema})
    # Simple
    for tmpl, vocab, schema in SIMPLE_TEMPLATES:
        for v in vocab:
            facts.append({"trigger": tmpl, "prompt": tmpl, "gold": " " + v, "schema": schema})
    return facts


def gen_user_facts_xxl(n=2000):
    """Sample n UNIQUE-TRIGGER facts. Each fact has a distinct trigger from
    the enumerated pool."""
    pool = gen_unique_triggers()
    random.Random(42).shuffle(pool)
    # Dedupe by trigger (different golds for same trigger collapse to one)
    seen = {}
    for f in pool:
        if f["trigger"] not in seen:
            seen[f["trigger"]] = f
        if len(seen) >= n:
            break
    return list(seen.values())[:n]


def gen_multi_users_xxl(n_users=100, facts_per_user=100):
    pool = gen_unique_triggers()
    users = []
    for u_id in range(n_users):
        rng = random.Random(u_id * 7919 + 17)
        # Each user gets a unique sample of facts (their own private set)
        user_facts_with_dup = []
        seen = {}
        # Sample with dedup
        candidates = list(pool); rng.shuffle(candidates)
        for f in candidates:
            if f["trigger"] in seen: continue
            seen[f["trigger"]] = f
            user_facts_with_dup.append(f)
            if len(user_facts_with_dup) >= facts_per_user:
                break
        users.append({"user_id": u_id, "facts": user_facts_with_dup})
    return users


def main():
    out_dir = Path(f"{UAE_ROOT}/data")
    out_dir.mkdir(parents=True, exist_ok=True)
    print("Building XXL corpus...")
    pool = gen_unique_triggers()
    print(f"  Total unique-trigger facts in pool: {len(pool):,}")
    distinct_triggers = len({f['trigger'] for f in pool})
    print(f"  Distinct triggers: {distinct_triggers:,}")
    user_facts = gen_user_facts_xxl(2000)
    print(f"  user_facts (2000): {len(user_facts):,} unique triggers")
    multi_users = gen_multi_users_xxl(n_users=100, facts_per_user=100)
    n_per_user = [len(u['facts']) for u in multi_users]
    print(f"  multi_users: 100 users, facts/user min={min(n_per_user)} max={max(n_per_user)} mean={sum(n_per_user)/len(n_per_user):.1f}")

    bundle = {
        "user_facts": user_facts,
        "org_facts": [],  # subsumed by name-bearing in this corpus; keep empty to satisfy reads
        "multi_users": multi_users,
    }
    out_path = out_dir / "corpora_xxl.json"
    with open(out_path, "w") as f:
        json.dump(bundle, f, indent=2)
    print(f"\nWrote {out_path}")
    print(f"Sample USER facts:")
    for f in user_facts[:5]:
        print(f"  {f['trigger']!r:60s} -> {f['gold']!r}")
    print(f"\nSample multi-user 0, first 3 facts:")
    for f in multi_users[0]['facts'][:3]:
        print(f"  {f['trigger']!r:60s} -> {f['gold']!r}")


if __name__ == "__main__":
    main()
