"""
Synthetic per-user fact-set generator (produces data/users).

A user is a typed attribute bundle plus derived lists:
- `facts`: atomic natural-language statements (verbalizations of attributes)
- `direct_qa`: (question, answer) pairs exercising a single fact (POLAR-style)
- `indirect_qa`: (question, answer, required_fact_keys) pairs exercising 2+ facts

Indirect question schemas (with programmatic ground truth):
  AGE        : current-year arithmetic from a birth year
  DAY_SET    : set membership / negation over a routine day list
  COMPARE    : which of two people is older / younger
  ROUTINE    : "what does user do on [day]" from weekly schedule
  ALLERGY    : given a dish, which ingredient must be avoided
  POLICY     : conditional lookup composed across facts
  COMMUTE    : simple numeric composition (commute mins + work start)

Facts are verbalized in multiple paraphrases (like POLAR's FACTUAL_STATEMENTS)
so the LoRA sees the same fact in several surface forms.
"""
from __future__ import annotations
import os
UAE_ROOT = os.environ.get("USER_AS_ENGRAM_ROOT") or (
    os.path.dirname(os.environ["NANOCHAT_BASE_DIR"]) if os.environ.get("NANOCHAT_BASE_DIR")
    else os.getcwd())


import json
import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

# Reference date anchoring age arithmetic.
NOW_YEAR = 2026

# ---------------------------------------------------------------------------
# Canonical vocabulary (kept deterministic by seed so users differ but schema
# is shared — essential for Stage C "learn to read any adapter" transfer).
# ---------------------------------------------------------------------------

FIRST_NAMES = [
    "Alex", "Jordan", "Taylor", "Morgan", "Casey", "Riley", "Quinn", "Avery",
    "Sam", "Drew", "Rowan", "Emery", "Finley", "Harper", "Kai", "Logan",
    "Maya", "Noa", "Parker", "Reese", "Sage", "Skyler", "Toby", "Wren",
]
LAST_NAMES = [
    "Chen", "Patel", "Garcia", "Nguyen", "Okafor", "Silva", "Ivanov", "Dubois",
    "Kovacs", "Brooks", "Ahmed", "Johansson", "Tanaka", "Weber", "Ramos", "Kim",
]
CITIES = [
    ("Seattle", "PST"), ("Austin", "CST"), ("Boston", "EST"), ("Denver", "MST"),
    ("Chicago", "CST"), ("Portland", "PST"), ("Raleigh", "EST"), ("Minneapolis", "CST"),
]
EMPLOYERS = [
    "Acme Logistics", "Helix Biotech", "Northwind Analytics", "Pine & Brass",
    "Orbit Media", "Cedar Robotics", "Lumen Capital", "Summit Paper Co",
]
JOBS = [
    "software engineer", "data analyst", "product designer", "nurse practitioner",
    "high-school teacher", "radiologist", "civil engineer", "paralegal",
]
PETS = [
    ("dog", "Biscuit"), ("dog", "Poppy"), ("cat", "Mochi"), ("cat", "Figaro"),
    ("rabbit", "Clover"), ("parrot", "Tango"), ("dog", "Hazel"), ("cat", "Ember"),
]
CARS = [
    ("2019 Honda Civic", 2019), ("2021 Toyota RAV4", 2021), ("2018 Subaru Outback", 2018),
    ("2022 Ford Maverick", 2022), ("2020 Mazda CX-5", 2020), ("2017 VW Golf", 2017),
]
HOBBIES = ["rock climbing", "oil painting", "bread baking", "trail running",
           "chess", "pottery", "woodworking", "birdwatching"]
CUISINES = ["Thai", "Italian", "Ethiopian", "Vietnamese", "Mexican", "Japanese"]
ALLERGIES = ["peanuts", "shellfish", "dairy", "gluten", "tree nuts", "eggs"]
# ingredient -> list of dishes that contain it (used for ALLERGY indirect Qs)
DISH_INGREDIENTS = {
    "peanuts":    ["pad thai", "kung pao chicken", "peanut noodles"],
    "shellfish":  ["shrimp pad thai", "lobster roll", "clam chowder"],
    "dairy":      ["alfredo pasta", "cheese pizza", "ice cream sundae"],
    "gluten":     ["spaghetti", "sourdough bread", "flour tortillas"],
    "tree nuts":  ["pesto pasta", "walnut brownies", "almond croissant"],
    "eggs":       ["carbonara", "quiche lorraine", "french toast"],
}
DAYS = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]

# Weekly-activity slots and anchor days (routines).
ACTIVITIES = [
    ("gym", ["Tuesday", "Thursday"]),
    ("gym", ["Monday", "Wednesday", "Friday"]),
    ("yoga", ["Sunday"]),
    ("book club", ["Wednesday"]),
    ("therapy", ["Friday"]),
    ("pickup basketball", ["Saturday"]),
]


# ---------------------------------------------------------------------------
# User dataclass
# ---------------------------------------------------------------------------

@dataclass
class User:
    uid: str
    attrs: dict[str, Any]
    facts: list[dict[str, Any]] = field(default_factory=list)
    direct_qa: list[dict[str, Any]] = field(default_factory=list)
    indirect_qa: list[dict[str, Any]] = field(default_factory=list)

    def to_json(self) -> dict[str, Any]:
        return {
            "uid": self.uid,
            "attrs": self.attrs,
            "facts": self.facts,
            "direct_qa": self.direct_qa,
            "indirect_qa": self.indirect_qa,
        }


# ---------------------------------------------------------------------------
# Attribute sampler
# ---------------------------------------------------------------------------

def _sample_attrs(rng: random.Random, uid: str) -> dict[str, Any]:
    first = rng.choice(FIRST_NAMES)
    last = rng.choice(LAST_NAMES)
    birth_year = rng.randint(1968, 2000)
    city, tz = rng.choice(CITIES)
    job = rng.choice(JOBS)
    employer = rng.choice(EMPLOYERS)
    workdays = rng.choice([
        ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday"],
        ["Monday", "Tuesday", "Wednesday", "Thursday"],
        ["Tuesday", "Wednesday", "Thursday", "Friday", "Saturday"],
    ])
    work_start = rng.choice(["08:30", "09:00", "09:30", "10:00"])
    commute_min = rng.choice([15, 20, 25, 30, 40, 45])
    pet_sp, pet_name = rng.choice(PETS)
    car_desc, car_year = rng.choice(CARS)
    hobby1, hobby2 = rng.sample(HOBBIES, 2)
    cuisine = rng.choice(CUISINES)
    allergy = rng.choice(ALLERGIES)
    # Spouse / child
    sp_first = rng.choice([n for n in FIRST_NAMES if n != first])
    sp_year = birth_year + rng.randint(-4, 4)
    has_child = rng.random() < 0.7
    child_first = rng.choice([n for n in FIRST_NAMES if n not in (first, sp_first)])
    child_year = rng.randint(2011, 2022) if has_child else None
    # Sibling (older/younger)
    sib_first = rng.choice([n for n in FIRST_NAMES if n not in (first, sp_first, child_first or "")])
    sib_year = birth_year + rng.choice([-3, -2, 2, 3, 4])
    # Weekly activity
    act_name, act_days = rng.choice(ACTIVITIES)
    # Budget
    rent = rng.choice([1400, 1800, 2200, 2600, 3100])
    # Emergency contact
    ec_first = rng.choice([n for n in FIRST_NAMES if n not in (first, sp_first)])
    ec_phone_last = f"{rng.randint(1000,9999)}"
    # Phone
    phone_model = rng.choice(["iPhone 15", "Pixel 9", "iPhone 14 Pro", "Pixel 8a"])
    # Doctor / dentist
    doctor = rng.choice(["Dr. Mendez", "Dr. Holloway", "Dr. Ortiz", "Dr. Krause"])
    dentist = rng.choice(["Dr. Yoon", "Dr. Bhatt", "Dr. Akeredolu"])
    # Coffee order
    coffee = rng.choice(["oat-milk latte", "cortado", "cold brew", "pour-over, no milk"])
    # Running pace (min/mile)
    pace = rng.choice([8, 9, 10, 11])
    # Shoe size
    shoe = rng.choice([7, 8, 9, 10, 11, 12])
    # Favorite color
    color = rng.choice(["forest green", "navy", "burnt orange", "charcoal", "slate"])

    return dict(
        uid=uid,
        first_name=first, last_name=last, full_name=f"{first} {last}",
        birth_year=birth_year,
        city=city, timezone=tz,
        job=job, employer=employer,
        workdays=workdays, work_start=work_start, commute_min=commute_min,
        pet_species=pet_sp, pet_name=pet_name,
        car_desc=car_desc, car_year=car_year,
        hobby1=hobby1, hobby2=hobby2,
        cuisine=cuisine, allergy=allergy,
        spouse_name=sp_first, spouse_year=sp_year,
        child_name=child_first if has_child else None,
        child_year=child_year,
        sibling_name=sib_first, sibling_year=sib_year,
        activity=act_name, activity_days=act_days,
        rent=rent,
        emergency_contact=ec_first, ec_phone_last=ec_phone_last,
        phone_model=phone_model,
        doctor=doctor, dentist=dentist,
        coffee=coffee, running_pace=pace, shoe_size=shoe, color=color,
    )


# ---------------------------------------------------------------------------
# Fact verbalizers. Each returns a list of paraphrases for one fact key.
# Keeping the key stable matters: direct eval matches against it.
# ---------------------------------------------------------------------------

def _verbalize(a: dict[str, Any]) -> list[dict[str, Any]]:
    u = a["full_name"]
    facts = []

    def F(key: str, answer: Any, paraphrases: list[str]):
        facts.append({"key": key, "answer": answer, "paraphrases": paraphrases})

    F("name", a["full_name"], [
        f"My name is {a['full_name']}.",
        f"I go by {a['first_name']} {a['last_name']}.",
        f"The user's full name is {a['full_name']}.",
    ])
    F("birth_year", a["birth_year"], [
        f"{u} was born in {a['birth_year']}.",
        f"I was born in {a['birth_year']}.",
        f"My birth year is {a['birth_year']}.",
    ])
    F("city", a["city"], [
        f"{u} lives in {a['city']}.",
        f"I live in {a['city']}.",
        f"Home city: {a['city']}.",
    ])
    F("timezone", a["timezone"], [
        f"{u}'s timezone is {a['timezone']}.",
        f"I'm in {a['timezone']}.",
    ])
    F("job", a["job"], [
        f"{u} works as a {a['job']}.",
        f"My job is {a['job']}.",
    ])
    F("employer", a["employer"], [
        f"{u} is employed by {a['employer']}.",
        f"I work at {a['employer']}.",
    ])
    workdays_str = ", ".join(a["workdays"])
    F("workdays", a["workdays"], [
        f"My working days are {workdays_str}.",
        f"{u}'s in-office days: {workdays_str}.",
        f"I work {workdays_str}.",
    ])
    F("work_start", a["work_start"], [
        f"My workday starts at {a['work_start']}.",
        f"Work start time: {a['work_start']}.",
    ])
    F("commute_min", a["commute_min"], [
        f"My commute is {a['commute_min']} minutes each way.",
        f"It takes {a['commute_min']} minutes to get to work.",
    ])
    F("pet", f"{a['pet_species']} named {a['pet_name']}", [
        f"I have a {a['pet_species']} named {a['pet_name']}.",
        f"{u}'s pet is a {a['pet_species']} named {a['pet_name']}.",
    ])
    F("car", a["car_desc"], [
        f"I drive a {a['car_desc']}.",
        f"{u}'s car is a {a['car_desc']}.",
    ])
    F("car_year", a["car_year"], [
        f"My car is a {a['car_year']} model.",
        f"The model year of my car is {a['car_year']}.",
    ])
    F("hobby1", a["hobby1"], [
        f"{u} enjoys {a['hobby1']}.",
        f"My main hobby is {a['hobby1']}.",
    ])
    F("hobby2", a["hobby2"], [
        f"I also enjoy {a['hobby2']}.",
        f"Another hobby of mine is {a['hobby2']}.",
    ])
    F("cuisine", a["cuisine"], [
        f"My favorite cuisine is {a['cuisine']} food.",
        f"{u} loves {a['cuisine']} food.",
    ])
    F("allergy", a["allergy"], [
        f"I have a {a['allergy']} allergy.",
        f"{u} is allergic to {a['allergy']}.",
        f"Allergy on file: {a['allergy']}.",
    ])
    F("spouse_name", a["spouse_name"], [
        f"My spouse is {a['spouse_name']}.",
        f"{u}'s partner is named {a['spouse_name']}.",
    ])
    F("spouse_year", a["spouse_year"], [
        f"{a['spouse_name']} was born in {a['spouse_year']}.",
        f"My spouse's birth year is {a['spouse_year']}.",
    ])
    if a["child_name"]:
        F("child_name", a["child_name"], [
            f"My child is named {a['child_name']}.",
            f"{u}'s child is {a['child_name']}.",
        ])
        F("child_year", a["child_year"], [
            f"{a['child_name']} was born in {a['child_year']}.",
            f"My child's birth year is {a['child_year']}.",
        ])
    F("sibling_name", a["sibling_name"], [
        f"My sibling is {a['sibling_name']}.",
        f"{u}'s sibling is named {a['sibling_name']}.",
    ])
    F("sibling_year", a["sibling_year"], [
        f"{a['sibling_name']} was born in {a['sibling_year']}.",
        f"My sibling's birth year is {a['sibling_year']}.",
    ])
    F("activity", a["activity"], [
        f"My weekly activity is {a['activity']}.",
        f"{u} does {a['activity']} each week.",
    ])
    F("activity_days", a["activity_days"], [
        f"I do {a['activity']} on {' and '.join(a['activity_days'])}.",
        f"{a['activity'].capitalize()} days: {', '.join(a['activity_days'])}.",
    ])
    F("rent", a["rent"], [
        f"My monthly rent is ${a['rent']}.",
        f"{u}'s rent is ${a['rent']} per month.",
    ])
    F("emergency_contact", a["emergency_contact"], [
        f"My emergency contact is {a['emergency_contact']}.",
        f"{u}'s emergency contact is named {a['emergency_contact']}.",
    ])
    F("ec_phone_last", a["ec_phone_last"], [
        f"{a['emergency_contact']}'s phone ends in {a['ec_phone_last']}.",
        f"Emergency contact phone (last 4): {a['ec_phone_last']}.",
    ])
    F("phone_model", a["phone_model"], [
        f"I use a {a['phone_model']}.",
        f"{u}'s phone is a {a['phone_model']}.",
    ])
    F("doctor", a["doctor"], [
        f"My primary doctor is {a['doctor']}.",
        f"{u}'s doctor is {a['doctor']}.",
    ])
    F("dentist", a["dentist"], [
        f"My dentist is {a['dentist']}.",
        f"{u} sees {a['dentist']} for dental care.",
    ])
    F("coffee", a["coffee"], [
        f"My usual coffee order is a {a['coffee']}.",
        f"{u}'s go-to coffee is a {a['coffee']}.",
    ])
    F("running_pace", a["running_pace"], [
        f"My running pace is about {a['running_pace']} minutes per mile.",
        f"{u} runs at ~{a['running_pace']} min/mile.",
    ])
    F("shoe_size", a["shoe_size"], [
        f"My shoe size is {a['shoe_size']}.",
        f"{u} wears a size {a['shoe_size']}.",
    ])
    F("color", a["color"], [
        f"My favorite color is {a['color']}.",
        f"{u}'s favorite color is {a['color']}.",
    ])
    return facts


# ---------------------------------------------------------------------------
# Direct QA (one per fact, POLAR-style)
# ---------------------------------------------------------------------------

def _direct_qa(a: dict[str, Any]) -> list[dict[str, Any]]:
    def Q(key, q, a_str):
        return {"key": key, "question": q, "answer": str(a_str)}
    q = []
    q.append(Q("name", "What is my full name?", a["full_name"]))
    q.append(Q("birth_year", "What year was I born?", a["birth_year"]))
    q.append(Q("city", "What city do I live in?", a["city"]))
    q.append(Q("timezone", "What timezone am I in?", a["timezone"]))
    q.append(Q("job", "What is my job?", a["job"]))
    q.append(Q("employer", "Who is my employer?", a["employer"]))
    q.append(Q("workdays", "Which days of the week do I work?", ", ".join(a["workdays"])))
    q.append(Q("work_start", "What time do I start work?", a["work_start"]))
    q.append(Q("commute_min", "How many minutes is my commute each way?", a["commute_min"]))
    q.append(Q("pet", "What pet do I have and what is its name?",
              f"a {a['pet_species']} named {a['pet_name']}"))
    q.append(Q("car", "What car do I drive?", a["car_desc"]))
    q.append(Q("car_year", "What is the model year of my car?", a["car_year"]))
    q.append(Q("hobby1", "What is my main hobby?", a["hobby1"]))
    q.append(Q("hobby2", "What other hobby do I have?", a["hobby2"]))
    q.append(Q("cuisine", "What is my favorite cuisine?", a["cuisine"]))
    q.append(Q("allergy", "What am I allergic to?", a["allergy"]))
    q.append(Q("spouse_name", "What is my spouse's name?", a["spouse_name"]))
    q.append(Q("spouse_year", "What year was my spouse born?", a["spouse_year"]))
    if a["child_name"]:
        q.append(Q("child_name", "What is my child's name?", a["child_name"]))
        q.append(Q("child_year", "What year was my child born?", a["child_year"]))
    q.append(Q("sibling_name", "What is my sibling's name?", a["sibling_name"]))
    q.append(Q("sibling_year", "What year was my sibling born?", a["sibling_year"]))
    q.append(Q("activity", "What weekly activity do I do?", a["activity"]))
    q.append(Q("activity_days", f"Which days of the week do I do {a['activity']}?",
              ", ".join(a["activity_days"])))
    q.append(Q("rent", "What is my monthly rent (in dollars)?", a["rent"]))
    q.append(Q("emergency_contact", "Who is my emergency contact?", a["emergency_contact"]))
    q.append(Q("ec_phone_last", "What are the last four digits of my emergency contact's phone number?",
              a["ec_phone_last"]))
    q.append(Q("phone_model", "What phone do I use?", a["phone_model"]))
    q.append(Q("doctor", "Who is my primary doctor?", a["doctor"]))
    q.append(Q("dentist", "Who is my dentist?", a["dentist"]))
    q.append(Q("coffee", "What is my usual coffee order?", a["coffee"]))
    q.append(Q("running_pace", "What is my running pace (min/mile)?", a["running_pace"]))
    q.append(Q("shoe_size", "What is my shoe size?", a["shoe_size"]))
    q.append(Q("color", "What is my favorite color?", a["color"]))
    return q


# ---------------------------------------------------------------------------
# Indirect QA — requires composing 2+ facts. Ground truth computed directly
# from `attrs`, so we know the answer without involving the model.
# ---------------------------------------------------------------------------

def _indirect_qa(a: dict[str, Any], rng: random.Random) -> list[dict[str, Any]]:
    """Expanded to ≥4 questions per schema so within-schema held split is
    clean (every schema has ≥2 train / ≥2 held after stratified 50/50).
    Each question carries a `variant` tag so build_programmatic_trace can
    dispatch deterministically on (schema, variant)."""
    out: list[dict[str, Any]] = []

    def add(schema: str, q: str, ans: Any, keys: list[str], variant: str):
        out.append({
            "schema": schema,
            "variant": variant,
            "question": q,
            "answer": str(ans),
            "required_fact_keys": keys,
        })

    # -------- AGE (4–5) --------
    age = NOW_YEAR - a["birth_year"]
    add("AGE", f"How old am I right now (it's {NOW_YEAR})?", age,
        ["birth_year"], "self")
    sp_age = NOW_YEAR - a["spouse_year"]
    add("AGE", f"How old is my spouse right now (it's {NOW_YEAR})?", sp_age,
        ["spouse_name", "spouse_year"], "spouse")
    if a["child_name"]:
        ch_age = NOW_YEAR - a["child_year"]
        add("AGE", f"How old is my child right now (it's {NOW_YEAR})?", ch_age,
            ["child_name", "child_year"], "child")
    sib_age = NOW_YEAR - a["sibling_year"]
    add("AGE", f"How old is my sibling right now (it's {NOW_YEAR})?", sib_age,
        ["sibling_name", "sibling_year"], "sibling")
    car_age = NOW_YEAR - a["car_year"]
    add("AGE", f"How old is my car right now (it's {NOW_YEAR})?", car_age,
        ["car", "car_year"], "car")

    # -------- COMPARE (4) --------
    def _older_of(na: str, ya: int, nb: str, yb: int) -> str:
        if ya < yb: return na
        if ya > yb: return nb
        return "same age"

    # me vs spouse
    add("COMPARE", "Who is older, me or my spouse?",
        _older_of("me", a["birth_year"], f"{a['spouse_name']} (my spouse)", a["spouse_year"]),
        ["birth_year", "spouse_name", "spouse_year"], "me_vs_spouse")
    # me vs sibling
    add("COMPARE", "Who is older, me or my sibling?",
        _older_of("me", a["birth_year"], f"{a['sibling_name']} (my sibling)", a["sibling_year"]),
        ["birth_year", "sibling_name", "sibling_year"], "me_vs_sibling")
    # oldest of three (me, spouse, sibling)
    trio = [("me", a["birth_year"]),
            (f"{a['spouse_name']} (my spouse)", a["spouse_year"]),
            (f"{a['sibling_name']} (my sibling)", a["sibling_year"])]
    oldest = min(trio, key=lambda x: x[1])[0]
    add("COMPARE", "Among me, my spouse, and my sibling, who was born earliest?",
        oldest,
        ["birth_year", "spouse_name", "spouse_year", "sibling_name", "sibling_year"],
        "oldest_of_three")
    # spouse vs sibling (kept from original)
    add("COMPARE", "Who is older, my spouse or my sibling?",
        _older_of(f"{a['spouse_name']} (spouse)", a["spouse_year"],
                  f"{a['sibling_name']} (sibling)", a["sibling_year"]),
        ["spouse_name", "spouse_year", "sibling_name", "sibling_year"],
        "spouse_vs_sibling")

    # -------- DAY_SET (4) --------
    is_sun_work = "Sunday" in a["workdays"]
    add("DAY_SET", "Do I work on Sundays?",
        "Yes" if is_sun_work else "No",
        ["workdays"], "work_sun")
    is_sat_work = "Saturday" in a["workdays"]
    add("DAY_SET", "Do I work on Saturdays?",
        "Yes" if is_sat_work else "No",
        ["workdays"], "work_sat")
    not_act_day = next(d for d in DAYS if d not in a["activity_days"])
    add("DAY_SET", f"Do I have {a['activity']} on {not_act_day}?", "No",
        ["activity", "activity_days"], "activity_off_day")
    act_day = a["activity_days"][0]
    add("DAY_SET", f"Do I have {a['activity']} on {act_day}?", "Yes",
        ["activity", "activity_days"], "activity_on_day")

    # -------- ROUTINE (4) --------
    day = a["activity_days"][0]
    add("ROUTINE", f"What scheduled weekly activity do I have on {day}?",
        a["activity"], ["activity", "activity_days"], "activity_on_day")
    add("ROUTINE", f"Which day(s) of the week do I do {a['activity']}?",
        ", ".join(a["activity_days"]),
        ["activity", "activity_days"], "days_of_activity")
    add("ROUTINE", f"How many days per week do I do {a['activity']}?",
        len(a["activity_days"]),
        ["activity", "activity_days"], "count_days")
    add("ROUTINE", "What is my recurring weekly activity?",
        a["activity"], ["activity"], "name_activity")

    # -------- ALLERGY (4) --------
    allergen_dishes = DISH_INGREDIENTS[a["allergy"]]
    safe_allergen = next(alg for alg in DISH_INGREDIENTS if alg != a["allergy"])
    safe_dishes = DISH_INGREDIENTS[safe_allergen]
    add("ALLERGY", f"Is it safe for me to eat {allergen_dishes[0]}?", "No",
        ["allergy"], "unsafe_0")
    add("ALLERGY", f"From the standpoint of my allergy alone, is {safe_dishes[0]} safe for me?",
        "Yes", ["allergy"], "safe_0")
    add("ALLERGY", f"Is it safe for me to eat {allergen_dishes[1]}?", "No",
        ["allergy"], "unsafe_1")
    add("ALLERGY", f"From the standpoint of my allergy alone, is {safe_dishes[1]} safe for me?",
        "Yes", ["allergy"], "safe_1")

    # -------- COMMUTE (4) --------
    hh, mm = map(int, a["work_start"].split(":"))
    start_min = hh * 60 + mm
    leave_min = start_min - a["commute_min"]
    lh, lm = divmod(leave_min, 60)
    arrive_min = start_min + a["commute_min"]
    ah, am = divmod(arrive_min, 60)
    add("COMMUTE",
        "If I want to arrive exactly at work start time, what time should I leave home?",
        f"{lh:02d}:{lm:02d}",
        ["work_start", "commute_min"], "leave_time")
    add("COMMUTE", "What is my round-trip commute time in minutes?",
        a["commute_min"] * 2,
        ["commute_min"], "round_trip")
    add("COMMUTE",
        "How many minutes per week do I spend commuting (round-trip on work days only)?",
        a["commute_min"] * 2 * len(a["workdays"]),
        ["commute_min", "workdays"], "weekly_minutes")
    add("COMMUTE",
        "If I leave home exactly at my usual work start time, when will I arrive at work?",
        f"{ah:02d}:{am:02d}",
        ["work_start", "commute_min"], "arrive_if_leave_at_start")

    # -------- POLICY (4) --------
    add("POLICY", "If I'm unreachable, who should a caller contact instead?",
        a["emergency_contact"], ["emergency_contact"], "emergency_contact")
    add("POLICY", "Who should I call about a medical issue?",
        a["doctor"], ["doctor"], "doctor")
    add("POLICY", "Who should I call about a dental issue?",
        a["dentist"], ["dentist"], "dentist")
    add("POLICY",
        "What are the last four digits of my emergency contact's phone number?",
        a["ec_phone_last"],
        ["emergency_contact", "ec_phone_last"], "ec_phone")

    # -------- MULTI (4) --------
    def _relation(delta: int) -> str:
        if delta == 0:
            return "the same age as me"
        if delta > 0:
            return f"{delta} year(s) younger than me"
        return f"{-delta} year(s) older than me"

    add("MULTI", "Is my sibling older or younger than me, and by how many years?",
        _relation(a["sibling_year"] - a["birth_year"]),
        ["birth_year", "sibling_year"], "sibling_delta")
    add("MULTI", "Is my spouse older or younger than me, and by how many years?",
        _relation(a["spouse_year"] - a["birth_year"]),
        ["birth_year", "spouse_year"], "spouse_delta")
    # spouse vs sibling delta
    d_ss = a["sibling_year"] - a["spouse_year"]
    if d_ss == 0:
        rel = "the same age"
    elif d_ss > 0:
        rel = f"my sibling is {d_ss} year(s) younger than my spouse"
    else:
        rel = f"my sibling is {-d_ss} year(s) older than my spouse"
    add("MULTI", "By how many years is my spouse older or younger than my sibling?",
        rel,
        ["spouse_year", "sibling_year"], "spouse_vs_sibling_delta")
    # future age in 10 years
    add("MULTI", f"How old will I be 10 years from now (it's {NOW_YEAR})?",
        age + 10,
        ["birth_year"], "future_age")

    return out


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def build_user(uid: str, seed: int) -> User:
    rng = random.Random(seed)
    attrs = _sample_attrs(rng, uid)
    facts = _verbalize(attrs)
    direct = _direct_qa(attrs)
    indirect = _indirect_qa(attrs, rng)
    return User(uid=uid, attrs=attrs, facts=facts,
                direct_qa=direct, indirect_qa=indirect)


def write_user(u: User, out_dir: Path) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{u.uid}.json"
    with open(path, "w") as f:
        json.dump(u.to_json(), f, indent=2)
    return path


def load_user(path: Path) -> User:
    d = json.loads(Path(path).read_text())
    return User(uid=d["uid"], attrs=d["attrs"],
                facts=d["facts"], direct_qa=d["direct_qa"],
                indirect_qa=d["indirect_qa"])


if __name__ == "__main__":
    out = Path(f"{UAE_ROOT}/data/users")
    for i in range(20):
        u = build_user(uid=f"u{i:03d}", seed=1000 + i)
        p = write_user(u, out)
        print(f"  wrote {p}  facts={len(u.facts)}  direct={len(u.direct_qa)}"
              f"  indirect={len(u.indirect_qa)}")
