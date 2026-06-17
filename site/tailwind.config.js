/** @type {import('tailwindcss').Config} */
export default {
  content: ["./index.html", "./src/**/*.{js,ts,jsx,tsx}"],
  theme: {
    extend: {
      colors: {
        // Warm parchment canvas + warm ink — print-adjacent, journal-like.
        paper: {
          DEFAULT: "#F4F0E6",
          50: "#FBF9F3",
          100: "#F7F4EA",
          200: "#EDE7D6",
          300: "#E2DAC4",
        },
        ink: {
          DEFAULT: "#211E18",
          50: "#6B6557",
          100: "#534D41",
          200: "#3A352C",
          300: "#2A261F",
          400: "#211E18",
        },
        // "ours" — the slate-blue from the paper.
        engram: {
          DEFAULT: "#34507F",
          light: "#5E79A8",
          lighter: "#8FA6C9",
          deep: "#23375C",
          wash: "#E7ECF3",
        },
        // baseline / cost — muted terracotta from the paper.
        rust: {
          DEFAULT: "#C24A3F",
          light: "#D6796F",
          deep: "#8F352D",
          wash: "#F3E3DF",
        },
        ochre: { DEFAULT: "#B8893B", light: "#D8BC7E" },
        rule: "#CFC6B2",
      },
      fontFamily: {
        display: ['"Fraunces Variable"', "Fraunces", "Georgia", "serif"],
        body: ['"Newsreader Variable"', "Newsreader", "Georgia", "serif"],
        mono: ['"IBM Plex Mono"', "ui-monospace", "SFMono-Regular", "monospace"],
      },
      letterSpacing: { tightest: "-0.04em" },
      maxWidth: { prose: "34rem" },
      keyframes: {
        grain: {
          "0%,100%": { transform: "translate(0,0)" },
          "10%": { transform: "translate(-3%,-2%)" },
          "30%": { transform: "translate(2%,-4%)" },
          "50%": { transform: "translate(-2%,3%)" },
          "70%": { transform: "translate(3%,2%)" },
          "90%": { transform: "translate(-3%,1%)" },
        },
      },
      animation: { grain: "grain 8s steps(6) infinite" },
    },
  },
  plugins: [],
};
