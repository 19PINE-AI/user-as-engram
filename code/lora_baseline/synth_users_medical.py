"""
Cross-schema test corpus for User-as-Engram (Tier 2 #6).

Schema: medical patient profile. Different surface forms than the personal
schema in synth_users.py, but parallel indirect-Q types (AGE-style arithmetic,
COMPARE, SET, ROUTINE, ALLERGY-equivalent) so the *meta-skill* is reusable
while *content* is fully different.

This lets us train shared LoRA on the personal schema and evaluate on the
medical schema, isolating whether the meta-skill transfers (a positive
result for the layered architecture) or is locked to schema surface forms
(a negative result, like Stage A's cross-schema collapse).
"""
from __future__ import annotations
import os
UAE_ROOT = os.environ.get("USER_AS_ENGRAM_ROOT") or (
    os.path.dirname(os.environ["NANOCHAT_BASE_DIR"]) if os.environ.get("NANOCHAT_BASE_DIR")
    else os.getcwd())
import json, random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

NOW_YEAR = 2026

# ---------------------------------------------------------------------------
# Medical-schema vocabulary
# ---------------------------------------------------------------------------

PATIENT_IDS = ["MRN-A0231", "MRN-B4781", "MRN-C9112", "MRN-D5034", "MRN-E2906",
               "MRN-F7823", "MRN-G1450", "MRN-H6071", "MRN-J3389", "MRN-K8245"]
SURNAMES = ["Chen", "Patel", "Garcia", "Nguyen", "Okafor", "Silva", "Ivanov",
            "Dubois", "Kovacs", "Brooks", "Ahmed", "Tanaka"]
FIRST_NAMES = ["Alex", "Jordan", "Taylor", "Morgan", "Casey", "Riley", "Quinn",
               "Avery", "Sam", "Drew", "Rowan", "Emery", "Finley", "Harper"]
CONDITIONS = ["type 2 diabetes", "hypertension", "asthma", "GERD",
              "hypothyroidism", "migraine", "atrial fibrillation"]
# medication -> typical dose mg
MEDS = [
    ("metformin", 1000), ("lisinopril", 20), ("levothyroxine", 100),
    ("atorvastatin", 40), ("amlodipine", 10), ("sertraline", 50),
    ("omeprazole", 20), ("ibuprofen", 400),
]
# drug class -> contraindicated medication examples (used for ALLERGY-equivalent)
DRUG_ALLERGY = {
    "penicillin":   ["amoxicillin", "ampicillin"],
    "sulfa":        ["sulfamethoxazole", "celecoxib"],
    "NSAIDs":       ["ibuprofen", "naproxen"],
    "statins":      ["atorvastatin", "simvastatin"],
    "aspirin":      ["aspirin"],
    "ACE-inhibitors":["lisinopril", "enalapril"],
}
SAFE_MEDS = ["acetaminophen", "amoxicillin", "loratadine", "diphenhydramine",
             "calcium carbonate", "ferrous sulfate"]
PHYSICIAN_NAMES = ["Dr. Patel", "Dr. Krause", "Dr. Anand", "Dr. Holm",
                    "Dr. Vasquez", "Dr. Akeredolu", "Dr. Larson", "Dr. Wu"]
CLINICS = ["Mt. Pine Clinic", "Lakeshore Family Practice", "Westwood Internal Medicine",
            "Cedar Cardiology Group", "Sunset Endocrine Associates"]
SPECIALTIES = ["primary care", "cardiology", "endocrinology", "pulmonology",
               "neurology", "gastroenterology"]
LAB_DAYS = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday"]
PHARMACIES = ["WalCare Pharmacy", "RxOne", "Cedar Drug", "Greenway Apothecary"]
BLOOD_TYPES = ["O+", "A+", "B+", "AB+", "O-", "A-"]


@dataclass
class Patient:
    uid: str
    attrs: dict
    facts: list
    direct_qa: list
    indirect_qa: list

    def to_json(self):
        return {"uid": self.uid, "attrs": self.attrs,
                 "facts": self.facts, "direct_qa": self.direct_qa,
                 "indirect_qa": self.indirect_qa}


# ---------------------------------------------------------------------------
# Sample one medical patient
# ---------------------------------------------------------------------------

def _sample_attrs(rng, uid):
    first = rng.choice(FIRST_NAMES)
    last = rng.choice(SURNAMES)
    birth_year = rng.randint(1950, 2000)
    diagnosis_year = birth_year + rng.randint(20, 50)
    diagnosis_year = min(diagnosis_year, NOW_YEAR - 1)
    return {
        "mrn": rng.choice(PATIENT_IDS),
        "first_name": first,
        "last_name": last,
        "full_name": f"{first} {last}",
        "birth_year": birth_year,
        "diagnosis_year": diagnosis_year,
        "primary_condition": rng.choice(CONDITIONS),
        "secondary_condition": rng.choice([c for c in CONDITIONS if c != "type 2 diabetes"]),
        "primary_med": rng.choice(MEDS),
        "secondary_med": rng.choice([m for m in MEDS if m[0] != "metformin"]),
        "drug_allergy": rng.choice(list(DRUG_ALLERGY.keys())),
        "primary_physician": rng.choice(PHYSICIAN_NAMES),
        "specialist": rng.choice(PHYSICIAN_NAMES),
        "specialty": rng.choice(SPECIALTIES),
        "clinic": rng.choice(CLINICS),
        "blood_type": rng.choice(BLOOD_TYPES),
        "weight_kg": rng.randint(55, 120),
        "height_cm": rng.randint(150, 195),
        "systolic": rng.randint(110, 145),
        "diastolic": rng.randint(65, 95),
        "lab_day": rng.choice(LAB_DAYS),
        "pharmacy": rng.choice(PHARMACIES),
        "follow_up_months": rng.choice([3, 6, 12]),
        "emergency_contact": rng.choice(FIRST_NAMES),
        "emergency_phone_last": f"{rng.randint(1000, 9999)}",
        "spouse_name": rng.choice([n for n in FIRST_NAMES if n != first]),
        "spouse_year": birth_year + rng.randint(-5, 5),
        "child_name": rng.choice(FIRST_NAMES),
        "child_year": birth_year + rng.randint(20, 35),
        "uid": uid,
    }


def _verbalize(a):
    pm = a["primary_med"]; sm = a["secondary_med"]
    return [
        {"key": "name", "answer": a["full_name"],
         "paraphrases": [f"The patient's name is {a['full_name']}.",
                          f"Patient: {a['full_name']}.",
                          f"Name on file: {a['full_name']}."]},
        {"key": "mrn", "answer": a["mrn"],
         "paraphrases": [f"The medical record number is {a['mrn']}.",
                          f"MRN: {a['mrn']}.",
                          f"Patient MRN is {a['mrn']}."]},
        {"key": "birth_year", "answer": a["birth_year"],
         "paraphrases": [f"The patient was born in {a['birth_year']}.",
                          f"Date of birth (year): {a['birth_year']}.",
                          f"Birth year: {a['birth_year']}."]},
        {"key": "diagnosis_year", "answer": a["diagnosis_year"],
         "paraphrases": [f"The primary condition was diagnosed in {a['diagnosis_year']}.",
                          f"Diagnosis year: {a['diagnosis_year']}."]},
        {"key": "primary_condition", "answer": a["primary_condition"],
         "paraphrases": [f"The patient's primary condition is {a['primary_condition']}.",
                          f"Primary diagnosis: {a['primary_condition']}.",
                          f"Main condition on chart: {a['primary_condition']}."]},
        {"key": "secondary_condition", "answer": a["secondary_condition"],
         "paraphrases": [f"A secondary condition is {a['secondary_condition']}.",
                          f"Secondary diagnosis: {a['secondary_condition']}."]},
        {"key": "primary_med", "answer": f"{pm[0]} {pm[1]} mg",
         "paraphrases": [f"The patient takes {pm[0]} {pm[1]} mg.",
                          f"Primary medication: {pm[0]} {pm[1]} mg.",
                          f"Daily prescription is {pm[0]} {pm[1]} mg."]},
        {"key": "primary_med_name", "answer": pm[0],
         "paraphrases": [f"The patient's primary medication is {pm[0]}.",
                          f"Main drug on the chart: {pm[0]}."]},
        {"key": "primary_med_dose", "answer": pm[1],
         "paraphrases": [f"The primary med is dosed at {pm[1]} mg.",
                          f"Daily dose of primary medication: {pm[1]} mg."]},
        {"key": "secondary_med_name", "answer": sm[0],
         "paraphrases": [f"A second medication on file is {sm[0]}.",
                          f"Secondary med: {sm[0]}."]},
        {"key": "drug_allergy", "answer": a["drug_allergy"],
         "paraphrases": [f"The patient is allergic to {a['drug_allergy']}.",
                          f"Drug allergy on file: {a['drug_allergy']}.",
                          f"Allergy: {a['drug_allergy']}."]},
        {"key": "primary_physician", "answer": a["primary_physician"],
         "paraphrases": [f"The primary care physician is {a['primary_physician']}.",
                          f"PCP: {a['primary_physician']}."]},
        {"key": "specialist", "answer": a["specialist"],
         "paraphrases": [f"The patient's specialist is {a['specialist']}.",
                          f"Specialist on chart: {a['specialist']}."]},
        {"key": "specialty", "answer": a["specialty"],
         "paraphrases": [f"The specialty involved is {a['specialty']}.",
                          f"Specialty: {a['specialty']}."]},
        {"key": "clinic", "answer": a["clinic"],
         "paraphrases": [f"The clinic is {a['clinic']}.",
                          f"Care location: {a['clinic']}."]},
        {"key": "blood_type", "answer": a["blood_type"],
         "paraphrases": [f"The patient's blood type is {a['blood_type']}.",
                          f"Blood type: {a['blood_type']}."]},
        {"key": "weight_kg", "answer": a["weight_kg"],
         "paraphrases": [f"The patient's weight is {a['weight_kg']} kg.",
                          f"Body weight: {a['weight_kg']} kg."]},
        {"key": "height_cm", "answer": a["height_cm"],
         "paraphrases": [f"The patient is {a['height_cm']} cm tall.",
                          f"Height on file: {a['height_cm']} cm."]},
        {"key": "systolic", "answer": a["systolic"],
         "paraphrases": [f"Systolic blood pressure: {a['systolic']} mmHg.",
                          f"The systolic reading is {a['systolic']}."]},
        {"key": "diastolic", "answer": a["diastolic"],
         "paraphrases": [f"Diastolic blood pressure: {a['diastolic']} mmHg.",
                          f"The diastolic reading is {a['diastolic']}."]},
        {"key": "lab_day", "answer": a["lab_day"],
         "paraphrases": [f"Lab work is scheduled on {a['lab_day']}.",
                          f"The patient does lab work on {a['lab_day']}."]},
        {"key": "pharmacy", "answer": a["pharmacy"],
         "paraphrases": [f"Prescriptions are filled at {a['pharmacy']}.",
                          f"Pharmacy of record: {a['pharmacy']}."]},
        {"key": "follow_up_months", "answer": a["follow_up_months"],
         "paraphrases": [f"Follow-up interval: {a['follow_up_months']} months.",
                          f"The patient returns every {a['follow_up_months']} months."]},
        {"key": "emergency_contact", "answer": a["emergency_contact"],
         "paraphrases": [f"Emergency contact name: {a['emergency_contact']}.",
                          f"In case of emergency, call {a['emergency_contact']}."]},
        {"key": "emergency_phone_last", "answer": a["emergency_phone_last"],
         "paraphrases": [f"Emergency contact phone ends in {a['emergency_phone_last']}.",
                          f"Last four digits of emergency phone: {a['emergency_phone_last']}."]},
        {"key": "spouse_name", "answer": a["spouse_name"],
         "paraphrases": [f"The patient's spouse is {a['spouse_name']}.",
                          f"Spouse on record: {a['spouse_name']}."]},
        {"key": "spouse_year", "answer": a["spouse_year"],
         "paraphrases": [f"The spouse was born in {a['spouse_year']}.",
                          f"Spouse's birth year is {a['spouse_year']}."]},
        {"key": "child_name", "answer": a["child_name"],
         "paraphrases": [f"The patient's child is {a['child_name']}.",
                          f"Child on record: {a['child_name']}."]},
        {"key": "child_year", "answer": a["child_year"],
         "paraphrases": [f"The child was born in {a['child_year']}.",
                          f"Child's birth year is {a['child_year']}."]},
    ]


def _direct_qa(a):
    pm = a["primary_med"]; sm = a["secondary_med"]
    return [
        ("name", "What is the patient's full name?", a["full_name"]),
        ("mrn", "What is the patient's MRN?", a["mrn"]),
        ("birth_year", "In what year was the patient born?", str(a["birth_year"])),
        ("diagnosis_year", "When was the patient's primary condition diagnosed (year)?", str(a["diagnosis_year"])),
        ("primary_condition", "What is the patient's primary condition?", a["primary_condition"]),
        ("secondary_condition", "What is a secondary condition?", a["secondary_condition"]),
        ("primary_med", "What is the patient's primary medication and dose?", f"{pm[0]} {pm[1]} mg"),
        ("primary_med_name", "What is the name of the patient's primary medication?", pm[0]),
        ("primary_med_dose", "What is the dose of the primary medication (mg)?", str(pm[1])),
        ("secondary_med_name", "What is a secondary medication name?", sm[0]),
        ("drug_allergy", "What drug class is the patient allergic to?", a["drug_allergy"]),
        ("primary_physician", "Who is the primary care physician?", a["primary_physician"]),
        ("specialist", "Who is the specialist?", a["specialist"]),
        ("specialty", "What specialty is involved in this patient's care?", a["specialty"]),
        ("clinic", "What clinic does the patient attend?", a["clinic"]),
        ("blood_type", "What is the patient's blood type?", a["blood_type"]),
        ("weight_kg", "What is the patient's weight in kilograms?", str(a["weight_kg"])),
        ("height_cm", "What is the patient's height in centimetres?", str(a["height_cm"])),
        ("systolic", "What is the patient's systolic blood pressure?", str(a["systolic"])),
        ("diastolic", "What is the patient's diastolic blood pressure?", str(a["diastolic"])),
        ("lab_day", "What day of the week does the patient do lab work?", a["lab_day"]),
        ("pharmacy", "Which pharmacy fills the prescriptions?", a["pharmacy"]),
        ("follow_up_months", "How many months apart are follow-up visits?", str(a["follow_up_months"])),
        ("emergency_contact", "Who is the patient's emergency contact?", a["emergency_contact"]),
        ("emergency_phone_last", "What are the last four digits of the emergency contact phone?", a["emergency_phone_last"]),
        ("spouse_name", "What is the spouse's name?", a["spouse_name"]),
        ("spouse_year", "In what year was the spouse born?", str(a["spouse_year"])),
        ("child_name", "What is the child's name?", a["child_name"]),
        ("child_year", "In what year was the child born?", str(a["child_year"])),
    ]


def _indirect_qa(a):
    """Parallel structure to personal schema: AGE, COMPARE, SET, ROUTINE,
    ALLERGY-equivalent, MULTI."""
    age_now = NOW_YEAR - a["birth_year"]
    spouse_age = NOW_YEAR - a["spouse_year"]
    child_age = NOW_YEAR - a["child_year"]
    years_since_dx = NOW_YEAR - a["diagnosis_year"]
    bmi = round(a["weight_kg"] / ((a["height_cm"]/100.0) ** 2), 1)
    pm_dose = a["primary_med"][1]
    annual_dose = pm_dose * 365
    contraindicated = DRUG_ALLERGY.get(a["drug_allergy"], [])

    out = [
        # AGE-style arithmetic
        ("AGE", "patient", "How old is the patient right now (it's 2026)?", str(age_now), ["birth_year"]),
        ("AGE", "spouse", "How old is the spouse right now (it's 2026)?", str(spouse_age), ["spouse_year"]),
        ("AGE", "child", "How old is the child right now (it's 2026)?", str(child_age), ["child_year"]),
        ("AGE", "years_since_dx", "How many years ago was the primary condition diagnosed (it's 2026)?", str(years_since_dx), ["diagnosis_year"]),
        ("AGE", "age_at_diagnosis", "How old was the patient at diagnosis?", str(a["diagnosis_year"] - a["birth_year"]), ["birth_year", "diagnosis_year"]),

        # COMPARE
        ("COMPARE", "patient_vs_spouse", "Is the patient older or younger than the spouse?", "older" if a["birth_year"] < a["spouse_year"] else "younger",
         ["birth_year", "spouse_year"]),
        ("COMPARE", "patient_vs_child", "Is the patient older or younger than the child?", "older" if a["birth_year"] < a["child_year"] else "younger",
         ["birth_year", "child_year"]),

        # SET (which day, lab day membership)
        ("SET", "lab_today", f"Is today a lab day for the patient (today is Wednesday)?",
         "Yes" if a["lab_day"] == "Wednesday" else "No", ["lab_day"]),
        ("SET", "lab_weekend", "Is the patient's lab day a weekend day?", "No", ["lab_day"]),

        # ROUTINE (next visit)
        ("ROUTINE", "next_visit_months", "How many months until the next follow-up visit?", str(a["follow_up_months"]),
         ["follow_up_months"]),
        ("ROUTINE", "visits_per_year", "How many follow-up visits does the patient have per year?",
         str(12 // a["follow_up_months"]), ["follow_up_months"]),

        # ALLERGY-equivalent (avoid contraindicated meds)
        *([("ALLERGY", f"avoid_{i}", f"Is {drug} safe to prescribe for this patient?",
            "No", ["drug_allergy"]) for i, drug in enumerate(contraindicated[:2])] if contraindicated else []),
        ("ALLERGY", "safe_acetaminophen", "Is acetaminophen safe to prescribe for this patient (assuming no other allergies)?",
         "Yes" if "acetaminophen" not in (contraindicated or []) else "No",
         ["drug_allergy"]),
        ("ALLERGY", "safe_amoxicillin", "Is amoxicillin safe to prescribe for this patient (assuming no other allergies)?",
         "Yes" if "amoxicillin" not in (contraindicated or []) else "No",
         ["drug_allergy"]),

        # MULTI (compose 2+ facts)
        ("MULTI", "annual_dose", "Total milligrams of primary medication taken per year (365 days)?",
         str(annual_dose), ["primary_med_dose"]),
        ("MULTI", "bmi", "What is the patient's BMI (rounded to one decimal)?",
         str(bmi), ["weight_kg", "height_cm"]),
        ("MULTI", "hypertensive", "Based on systolic and diastolic, is the patient hypertensive (systolic >=140 OR diastolic >=90)?",
         "Yes" if (a["systolic"] >= 140 or a["diastolic"] >= 90) else "No", ["systolic", "diastolic"]),
        ("MULTI", "spouse_delta", "Is the spouse older or younger than the patient, and by how many years?",
         f"{abs(a['spouse_year'] - a['birth_year'])} year(s) " +
            ("older" if a["spouse_year"] < a["birth_year"] else "younger"),
         ["birth_year", "spouse_year"]),
        ("MULTI", "ec_phone_last", "What are the last four digits of the emergency contact's phone?",
         a["emergency_phone_last"], ["emergency_contact", "emergency_phone_last"]),
    ]
    return out


def build_patient(uid, seed):
    rng = random.Random(seed)
    a = _sample_attrs(rng, uid)
    facts = _verbalize(a)
    direct = [{"key": k, "question": q, "answer": ans} for (k, q, ans) in _direct_qa(a)]
    indirect = [{"schema": s, "variant": v, "question": q, "answer": ans,
                  "required_fact_keys": rfk}
                 for (s, v, q, ans, rfk) in _indirect_qa(a)]
    return Patient(uid=uid, attrs=a, facts=facts, direct_qa=direct, indirect_qa=indirect)


if __name__ == "__main__":
    out_dir = Path(f"{UAE_ROOT}/data/users_medical")
    out_dir.mkdir(parents=True, exist_ok=True)
    for i in range(20):
        uid = f"m{i:03d}"   # m for medical
        p = build_patient(uid, seed=2000 + i)
        path = out_dir / f"{uid}.json"
        with open(path, "w") as f:
            json.dump(p.to_json(), f, indent=2)
        print(f"  wrote {path}  facts={len(p.facts)}  direct={len(p.direct_qa)}  indirect={len(p.indirect_qa)}")
