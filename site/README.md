# User as Engram — interactive site

A visual-first scrollytelling site for the paper. The idea is explained through
bespoke SVG visuals and interactions, not transcribed prose — one idea per
screen. React + Vite + Tailwind + [motion](https://motion.dev), with self-hosted
fonts (Fraunces / Newsreader / IBM Plex Mono). No chart library; every visual is
hand-rolled SVG/Canvas.

## Run

```bash
cd site
npm install
npm run dev        # → http://localhost:5173/research/user-as-engram/
npm run build      # → dist/   (static, self-contained)
npm run preview    # serve dist/ locally
```

Deployed under `…/research/user-as-engram/` (see `vite.config.ts` `base`).

## Structure

```
src/
├── App.tsx                 # the scene sequence (single-page scrollytelling)
├── main.tsx                # font imports + mount
├── index.css               # design system (parchment/slate/rust, grain, type)
├── theme.ts                # palette constants for SVG fills
├── data/                   # SINGLE SOURCE OF TRUTH — transcribed from the paper
│   ├── headline.ts         #   canonical numbers (5.6×/7.4×, 33,000×, 88 KB, …)
│   ├── conditions.ts       #   the six conditions (descriptive names)
│   ├── ragScale.ts         #   KB-size sweep (layered-flat vs RAG-decay)
│   └── facts.ts            #   Maya's facts for the write/two-jobs demos
├── lib/                     # useInView / useScrollProgress, toy hash for the write demo
└── components/
    ├── layout/  Nav · Scene · Reveal
    ├── ui/      Counter (count-up)
    ├── scenes/  Hero · Resources
    └── visuals/ MemoryGrid · TwoJobs · PriceLocator · BrainSplit ·
                 ContaminationSplit · WriteAFact · GlassBox ·
                 LayeredBars · KBCrossover · ServingFlow
```

## The scenes

Hero → Two jobs → **Where the price lands** (thesis) → Brain split →
**Contamination** (Engram vs LoRA heatmaps) → **Write a fact** → **Glass box**
(gate / value-path / depth slider) → Layered payoff → **KB crossover** slider →
Serving → Resources.

## Keeping data fresh

All numbers live in `src/data/*.ts`, transcribed from the current paper so the
site cannot drift. If a paper number changes, update the relevant `data/` module
(not the components). Reduced-motion is respected throughout.
