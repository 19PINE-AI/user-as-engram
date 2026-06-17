// Palette mirrored from tailwind.config.js, for use in SVG/Canvas fills.
export const C = {
  paper: "#F4F0E6",
  paper50: "#FBF9F3",
  paper200: "#EDE7D6",
  ink: "#211E18",
  ink50: "#6B6557",
  rule: "#CFC6B2",
  engram: "#34507F",
  engramLight: "#5E79A8",
  engramLighter: "#8FA6C9",
  engramDeep: "#23375C",
  engramWash: "#E7ECF3",
  rust: "#C24A3F",
  rustLight: "#D6796F",
  rustWash: "#F3E3DF",
  ochre: "#B8893B",
} as const;

export const EASE = [0.16, 1, 0.3, 1] as const; // expo-out, used across reveals
