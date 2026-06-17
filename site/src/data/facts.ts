// Maya's facts — the running persona. Used by the interactive write demo and
// the two-jobs scene. Each fact has a trigger (suffix N-gram) and an answer.
export type Fact = {
  id: string;
  trigger: string; // what the user says / the stored prefix
  answer: string; // the value written into the rows
  kind: "recall" | "reason"; // which "job" a question about it tends to need
};

export const mayaFacts: Fact[] = [
  { id: "cardiologist", trigger: "My cardiologist is Dr.", answer: "Elena Vasquez", kind: "recall" },
  { id: "clinic", trigger: "Dr. Vasquez practises at", answer: "Lakeside Cardiology", kind: "recall" },
  { id: "allergy", trigger: "I am severely allergic to", answer: "penicillin", kind: "reason" },
  { id: "diet", trigger: "Since 2024 I have been", answer: "vegetarian", kind: "reason" },
];

// Two-jobs scene: a direct question vs an indirect one over the same facts.
export const questions = {
  recall: {
    q: "Who is my cardiologist?",
    needs: "Look the fact up.",
  },
  reason: {
    q: "I'm visiting my daughter in California next month — where should I go for a heart check-up?",
    needs: "Chain several facts and reason.",
  },
};
