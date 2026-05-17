"""
Generate large-scale fact corpora for User-as-Engram evaluation.

Two corpora:
  1) 1K USER + 1K ORG single-user facts (E1)
  2) 100 users × 100 facts (E2)

To support 100 unique trigger schemas per user, we expand to ~100
distinct schemas (personal attributes, relationships, preferences,
schedule, etc.) and large per-attribute pools (50–200 values each).

Saved to /home/ubuntu/user-as-engram/data/corpora_xl.json
"""
import json
import random
from pathlib import Path

random.seed(20260505)


# ===== Pool definitions (large enough for thousands of distinct values) =====
NAMES = ["Alex","Sam","Jordan","Casey","Morgan","Riley","Kai","Rowan","Quinn","Sage",
         "Avery","Skyler","Dakota","Phoenix","Nova","Eden","Reese","Hayden","Parker","Emerson",
         "Drew","Lane","Sloan","Tate","Elliot","Finley","Harper","Indigo","Jace","Kendall",
         "Logan","Marlowe","Noor","Ocean","Presley","Quincy","Robin","Saylor","Tatum","Uri",
         "Vesper","Wren","Xeno","Yael","Zane","Bell","Cleo","Dakari","Echo","Fable",
         "Gable","Hollis","Iver","Juno","Kelby","Lior","Maven","Nico","Onyx","Pax",
         "Quill","Reign","Sable","Thorne","Umber","Vale","West","Xander","Yarrow","Zion",
         "Arden","Briar","Cypress","Dune","Ember","Fox","Gale","Haven","Iris","Jet",
         "Kit","Lark","Moss","North","Oak","Pine","Ridge","Slate","Teal","Vega",
         "Wolf","Yew","Zephyr","Aspen","Brook","Cedar","Dell","Elm","Fern","Glen"]
SURNAMES = ["Patel","Kim","Garcia","Diaz","Singh","Lopez","Martin","Brown","Lee","Wilson",
            "Moore","Taylor","Anderson","Thomas","Jackson","White","Harris","Clark","Lewis","Walker",
            "Hall","Allen","Young","King","Wright","Hill","Scott","Green","Adams","Baker",
            "Nelson","Carter","Perez","Roberts","Turner","Phillips","Campbell","Parker","Evans","Edwards",
            "Collins","Stewart","Sanchez","Morris","Rogers","Reed","Cook","Bell","Murphy","Bailey",
            "Rivera","Cooper","Richardson","Cox","Howard","Ward","Brooks","Watson","Sanders","Price"]
CITIES = ["Portland","Seattle","Boston","Austin","Denver","Phoenix","Miami","Atlanta","Boise","Madison",
          "Tampa","Reno","Tulsa","Omaha","Tucson","Spokane","Eugene","Burlington","Olympia","Helena",
          "Nashville","Memphis","Houston","Dallas","Detroit","Cleveland","Pittsburgh","Charlotte","Raleigh","Albany"]
SPICES = ["saffron","cardamom","turmeric","cumin","fennel","clove","nutmeg","paprika","ginger","cinnamon",
          "anise","sumac","oregano","basil","thyme","sage","mint","rosemary","tarragon","chive"]
COLORS = ["charcoal","amber","crimson","emerald","ivory","sapphire","jade","russet","ochre","cobalt",
          "scarlet","violet","teal","magenta","indigo","gold","silver","copper","bronze","mint"]
PETS = ["cat","dog","rabbit","parrot","tortoise","ferret","gecko","hamster","goldfish","cockatoo"]
DAYS = ["Monday","Tuesday","Wednesday","Thursday","Friday","Saturday","Sunday"]
JOBS = ["engineer","teacher","nurse","architect","accountant","barista","journalist","chef",
         "dentist","lawyer","writer","designer","analyst","programmer","scientist","pilot"]
COMPANIES = ["Globex","Initech","Hooli","Stark","Wayne","Acme","Umbrella","Vandelay","Dunder","Soylent",
             "Pied Piper","Massive Dynamic","Cyberdyne","Tyrell","Weyland","Yoyodyne","Aperture","Buy n Large","Initrode","Krusty",
             "Rekall","Sirius Cybernetics","Nakatomi","Spacely","Cogswell","Lunar","Helios","Atlas","Apex","Nova Corp"]
BRANDS = ["Subaru","Toyota","Honda","Volvo","Mazda","Hyundai","Kia","Ford","Chevrolet","BMW",
          "Audi","Mercedes","Lexus","Acura","Tesla","Volkswagen","Nissan","Mitsubishi","Porsche","Jaguar"]
HOURS = ["7","8","9","10","11"]
MONTHS = ["January","February","March","April","May","June","July","August","September","October","November","December"]
DEPTS = ["support","sales","billing","legal","engineering","finance","operations","marketing","HR","procurement"]
ANIMALS = ["otter","owl","fox","bear","wolf","raven","lynx","elk","hawk","eagle","panda","tiger","lion","heron","stork"]
ALLERGIES = ["peanuts","shellfish","eggs","dairy","gluten","sesame","soy","tree-nuts","sulphites","mustard"]
HOBBIES = ["chess","painting","running","cycling","cooking","gardening","photography","reading","hiking","yoga"]
INSTRUMENTS = ["piano","guitar","violin","drums","saxophone","flute","cello","trumpet","harp","clarinet"]
LANGS = ["Spanish","French","German","Mandarin","Japanese","Korean","Italian","Portuguese","Russian","Arabic"]
SPORTS = ["tennis","basketball","soccer","baseball","football","golf","swimming","climbing","skiing","cycling"]
DRINKS = ["coffee","tea","matcha","kombucha","cider","ale","stout","lager","cabernet","chardonnay"]
SCHOOLS = ["Riverdale","Northside","Eastfield","Westbrook","Oakwood","Maplewood","Pinecrest","Hillside","Brookfield","Lakeside"]
COUNTRIES = ["Canada","France","Japan","Germany","Mexico","Italy","Spain","Brazil","India","Australia"]
EMOJIS = ["heart","star","moon","sun","cloud","tree","flower","wave","mountain","bird"]


# ===== USER schemas: 50+ patterns =====
USER_SCHEMA = [
    # Personal attributes
    ("My doctor's name is Dr.",                            SURNAMES,   "doctor"),
    ("My dentist's name is Dr.",                            SURNAMES,   "dentist"),
    ("My pediatrician's name is Dr.",                       SURNAMES,   "pediatrician"),
    ("My therapist's name is Dr.",                          SURNAMES,   "therapist"),
    ("My optometrist's name is Dr.",                        SURNAMES,   "optometrist"),
    ("My chiropractor's name is Dr.",                       SURNAMES,   "chiropractor"),
    ("My favorite spice is",                                SPICES,     "spice"),
    ("My pet is a",                                         PETS,       "pet"),
    ("My favorite color is",                                COLORS,     "color"),
    ("I live in",                                           CITIES,     "city"),
    ("I was raised in",                                     CITIES,     "raised_in"),
    ("My gym day is",                                       DAYS,       "gym_day"),
    ("My yoga day is",                                      DAYS,       "yoga_day"),
    ("My swim day is",                                      DAYS,       "swim_day"),
    ("My run day is",                                       DAYS,       "run_day"),
    ("My favorite hobby is",                                HOBBIES,    "hobby"),
    ("My instrument is the",                                INSTRUMENTS,"instrument"),
    ("I drive a",                                           BRANDS,     "car_brand"),
    # Relationships (first names)
    ("My emergency contact is my sibling",                  NAMES,      "sibling"),
    ("My spouse is",                                        NAMES,      "spouse"),
    ("My partner is",                                       NAMES,      "partner"),
    ("My best friend is",                                   NAMES,      "best_friend"),
    ("My cousin is",                                        NAMES,      "cousin"),
    ("My older brother is",                                 NAMES,      "older_brother"),
    ("My younger sister is",                                NAMES,      "younger_sister"),
    ("My nephew is",                                        NAMES,      "nephew"),
    ("My niece is",                                         NAMES,      "niece"),
    ("My boss is",                                          NAMES,      "boss"),
    ("My neighbor is",                                      NAMES,      "neighbor"),
    ("My roommate is",                                      NAMES,      "roommate"),
    ("My team lead is",                                     NAMES,      "team_lead"),
    # Preferences
    ("My favorite drink is",                                DRINKS,     "drink"),
    ("My favorite sport is",                                SPORTS,     "sport"),
    ("My favorite language is",                             LANGS,      "language"),
    ("My favorite season is",                               ["Spring","Summer","Fall","Winter"], "season"),
    ("My favorite weekday is",                              DAYS,       "fav_weekday"),
    ("My favorite cuisine is",                              COUNTRIES,  "cuisine_origin"),
    # Allergies
    ("I am allergic to",                                    ALLERGIES,  "allergy"),
    # Schedule
    ("I usually wake up at",                                HOURS,      "wake_hour"),
    # Education and history
    ("I went to high school at",                            SCHOOLS,    "high_school"),
    ("I studied in",                                        COUNTRIES,  "study_country"),
    # Work
    ("I work as a",                                         JOBS,       "job"),
    ("I work at",                                           COMPANIES,  "employer"),
    # Emoji-style preferences (unusual)
    ("My favorite symbol is the",                           EMOJIS,     "symbol"),
    # Other
    ("My favorite animal is the",                           ANIMALS,    "fav_animal"),
    ("My superstition involves a",                          ANIMALS,    "superstition_animal"),
    ("My weekly grocery day is",                            DAYS,       "grocery_day"),
    ("My laundry day is",                                   DAYS,       "laundry_day"),
    ("My favorite month is",                                MONTHS,     "fav_month"),
    ("My birth month is",                                   MONTHS,     "birth_month"),
    ("My anniversary month is",                             MONTHS,     "anniversary_month"),
    ("I prefer the city of",                                CITIES,     "preferred_city"),
    ("I avoid the spice",                                   SPICES,     "avoid_spice"),
    ("My emergency code is the color",                      COLORS,     "code_color"),
    ("My target hobby is",                                  HOBBIES,    "target_hobby"),
    # More relationships
    ("My godmother is",                                     NAMES,      "godmother"),
    ("My godfather is",                                     NAMES,      "godfather"),
    ("My mentor is",                                        NAMES,      "mentor"),
    ("My pen pal is",                                       NAMES,      "pen_pal"),
    ("My personal trainer is",                              NAMES,      "trainer"),
    ("My accountant is",                                    NAMES,      "accountant"),
    ("My financial advisor is",                             NAMES,      "financial_advisor"),
    # More preferences
    ("My favorite tea is",                                  DRINKS,     "tea"),
    ("My favorite vegetable is",                            ["broccoli","spinach","kale","carrot","beet","corn","pea","pepper"], "vegetable"),
    ("My favorite fruit is",                                ["apple","banana","mango","papaya","kiwi","peach","plum","fig"], "fruit"),
    ("My favorite dessert is",                              ["cheesecake","brownie","cookie","tart","pudding","sorbet","cannoli","baklava"], "dessert"),
    ("My favorite breakfast is",                            ["oatmeal","granola","pancakes","waffles","crepes","yogurt","muffin","scone"], "breakfast"),
    ("My favorite holiday is",                              ["Halloween","Diwali","Holi","Hanukkah","Easter","Lunar","Solstice","Equinox"], "holiday"),
    # Misc
    ("My passport country is",                              COUNTRIES,  "passport"),
    ("My second language is",                               LANGS,      "second_lang"),
    ("My commute mode is",                                  ["bus","train","car","bike","walk","scooter","subway","ferry"], "commute_mode"),
    ("My preferred grocery store is",                       COMPANIES,  "grocery_store"),
    ("My favorite museum is in",                            CITIES,     "museum_city"),
    ("My favorite team is from",                            CITIES,     "team_city"),
    ("My favorite genre is",                                ["jazz","rock","classical","ambient","folk","techno","blues","reggae"], "music_genre"),
    ("My favorite poet is",                                 NAMES,      "poet"),
    ("My favorite director is",                             NAMES,      "director"),
    ("My favorite painter is",                              NAMES,      "painter"),
    ("My recurring dream involves a",                       ANIMALS,    "dream_animal"),
    ("My desk plant is a",                                  ["fern","cactus","succulent","orchid","aloe","ivy","palm","bamboo"], "desk_plant"),
    ("My ringtone uses the instrument",                     INSTRUMENTS,"ringtone_instrument"),
    ("My emergency phrase is the color",                    COLORS,     "emergency_phrase_color"),
    ("My security question name is",                        NAMES,      "security_q_name"),
]


# ===== ORG schemas: 30+ patterns =====
ORG_SCHEMA = [
    ("{O} office hours start at",                          HOURS,    "office_open"),
    ("{O} IT support extension is",                        ["1100","2200","3300","4400","5500","6600","7700","8800","9900"], "it_ext"),
    ("{O}'s mascot animal is the",                         ANIMALS,  "mascot"),
    ("{O} headquarters is in",                             CITIES,   "hq"),
    ("{O} CEO emeritus is",                                NAMES,    "ceo_emeritus"),
    ("{O}'s fiscal year starts in",                        MONTHS,   "fy_start"),
    ("{O}'s customer support email starts with",           DEPTS,    "support_email"),
    ("{O}'s monthly all-hands is on the first",            DAYS,     "allhands_day"),
    ("{O}'s flagship product is called",                   ANIMALS,  "flagship"),
    ("{O}'s parent company is",                            COMPANIES,"parent"),
    ("{O}'s training day is",                              DAYS,     "training_day"),
    ("{O}'s data center is in",                            CITIES,   "datacenter"),
    ("{O} was founded in the year",                        ["1985","1990","1995","2000","2005","2010","2015","2020"], "founded_year"),
    ("{O}'s main color is",                                COLORS,   "main_color"),
    ("{O}'s breakfast is catered by",                      COMPANIES,"breakfast_caterer"),
    ("{O}'s annual retreat is in",                         CITIES,   "retreat_city"),
    ("{O}'s quarterly review is in",                       MONTHS,   "qr_month"),
    ("{O} primary language is",                            LANGS,    "primary_lang"),
    ("{O}'s preferred shipping carrier is",                COMPANIES,"shipper"),
    ("{O}'s sister company is",                            COMPANIES,"sister"),
    ("{O}'s legal counsel is",                             COMPANIES,"counsel"),
    ("{O}'s tagline starts with the word",                 ["Empowering","Building","Connecting","Inspiring","Delivering"], "tagline_start"),
    ("{O}'s annual conference takes place in",             CITIES,   "conf_city"),
    ("{O}'s holiday week is in",                           MONTHS,   "holiday_month"),
    ("{O}'s code repository is named",                     ANIMALS,  "repo_name"),
    ("{O}'s standup is every",                             DAYS,     "standup_day"),
    ("{O}'s remote-work day is",                           DAYS,     "remote_day"),
    ("{O}'s annual hackathon theme is",                    EMOJIS,   "hack_theme"),
    ("{O}'s business model is",                            ["B2B","B2C","D2C","SaaS","PaaS"], "biz_model"),
    ("{O}'s primary industry is",                          ["fintech","biotech","retail","education","logistics","gaming","media","energy"], "industry"),
]


def gen_user_facts(n=1000):
    """Generate n single-user facts (mix across schemas)."""
    facts = []
    for i in range(n):
        rng = random.Random(i * 991 + 7)
        trig, vocab, schema = rng.choice(USER_SCHEMA)
        gold = rng.choice(vocab)
        facts.append({"trigger": trig, "prompt": trig, "gold": " " + gold, "schema": schema})
    return facts


def gen_org_facts(n=1000):
    facts = []
    for i in range(n):
        rng = random.Random(i * 1009 + 3)
        org = rng.choice(COMPANIES)
        trig_template, vocab, schema = rng.choice(ORG_SCHEMA)
        gold = rng.choice(vocab)
        trig = trig_template.format(O=org)
        facts.append({"trigger": trig, "prompt": trig, "gold": " " + gold, "org": org, "schema": schema})
    return facts


def gen_multi_users(n_users=100, facts_per_user=100):
    """Each user has facts_per_user facts with DISTINCT triggers (after dedup).

    With 50 USER schemas, having 100 distinct triggers per user is impossible.
    We expand by combining USER + ORG schemas (treating org-instantiated triggers
    as user-specific facts). Total schemas available: ~85 unique trigger templates.
    For 100 facts/user we may end up with ~85 unique triggers; we pad with
    repeated schemas using different golds — these are dropped at the dedupe-by-
    trigger step. So effective per-user fact count ≈ unique-template count.
    """
    users = []
    # Build flat list of (template_string, vocab_list, schema_id) where template
    # is a fully-resolved string (org name baked in if applicable, with placeholder).
    # For multi-user, keep ORG triggers parameterised with a per-user company.
    schema_pool = []
    for trig, vocab, schema in USER_SCHEMA:
        schema_pool.append(("user", trig, vocab, schema))
    # Org triggers: pick one canonical org per (user, schema) — different users
    # may have different org for the same schema.
    for trig_t, vocab, schema in ORG_SCHEMA:
        schema_pool.append(("org", trig_t, vocab, schema))

    for u_id in range(n_users):
        rng = random.Random(u_id * 7919 + 17)
        # Order schemas randomly for this user
        schemas = list(schema_pool)
        rng.shuffle(schemas)
        # Sample up to facts_per_user; dedupe by resolved trigger
        seen_triggers = set()
        facts = []
        for kind, trig_t, vocab, schema in schemas:
            if len(facts) >= facts_per_user: break
            if kind == "org":
                # pick a per-user company for this trigger
                org = rng.choice(COMPANIES)
                trigger = trig_t.format(O=org)
            else:
                trigger = trig_t
            if trigger in seen_triggers:
                continue
            seen_triggers.add(trigger)
            gold = rng.choice(vocab)
            facts.append({"trigger": trigger, "prompt": trigger, "gold": " " + gold,
                          "schema": schema, "kind": kind})
        users.append({"user_id": u_id, "facts": facts})
    return users


def main():
    out_dir = Path("/home/ubuntu/user-as-engram/data")
    out_dir.mkdir(parents=True, exist_ok=True)
    user_facts = gen_user_facts(1000)
    org_facts = gen_org_facts(1000)
    multi_users = gen_multi_users(n_users=100, facts_per_user=100)
    bundle = {
        "user_facts": user_facts,
        "org_facts": org_facts,
        "multi_users": multi_users,
    }
    out_path = out_dir / "corpora_xl.json"
    with open(out_path, "w") as f:
        json.dump(bundle, f, indent=2)
    n_unique_per_user = [len(u['facts']) for u in multi_users]
    print(f"Wrote {len(user_facts)} user facts, {len(org_facts)} org facts to {out_path}")
    print(f"Multi-users: {len(multi_users)} users")
    print(f"  per-user fact count: min={min(n_unique_per_user)} max={max(n_unique_per_user)} "
          f"mean={sum(n_unique_per_user)/len(n_unique_per_user):.1f}")
    # Sample
    print("\nSample USER fact:")
    print(f"  {user_facts[0]['trigger']!r:55s} -> {user_facts[0]['gold']!r}")
    print("\nSample ORG fact:")
    print(f"  {org_facts[0]['trigger']!r:55s} -> {org_facts[0]['gold']!r}")
    print(f"\nSample multi-user 0, first 3 facts:")
    for f in multi_users[0]['facts'][:3]:
        print(f"  [{f['kind']}] {f['trigger']!r:55s} -> {f['gold']!r}")


if __name__ == "__main__":
    main()
