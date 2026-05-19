# User as Engram — interactive paper site

React + Vite + Tailwind + Recharts. All result JSONs and figures from the
paper are bundled in `public/` and the app reads them at runtime.

## Run

```bash
cd site
npm install
npm run dev        # → http://localhost:5173
```

Build for static hosting:
```bash
npm run build      # → dist/
npm run preview    # serve dist/ locally
```

## Layout

```
site/
├── public/
│   ├── data/             # result JSONs (copied from ../results)
│   └── figs/             # PNG previews of paper figures
└── src/
    ├── App.tsx           # page composition + sidebar nav
    ├── index.css         # Tailwind base + theme tokens
    ├── types.ts          # shared TS types for data shapes
    ├── lib/data.ts       # JSON loaders + view-model builders
    └── components/
        ├── Hero.tsx                  # title, authors, stat chips
        ├── Abstract.tsx              # paper abstract
        ├── KeyStats.tsx              # 4 headline cards
        ├── Architecture.tsx          # interactive SVG diagram
        ├── LayeredHeadline.tsx       # Table 24 (A–J) sortable
        ├── ComparisonTableSection.tsx# All methods, family chips, sortable
        ├── KBScaleSection.tsx        # Trend chart + KB slider snapshot
        ├── ParetoSection.tsx         # Toggleable-family scatter
        ├── MultihopSection.tsx       # 8-pair RAG vs Engram
        ├── DataBrowser.tsx           # Per-user drill-down for all datasets
        ├── CitationCard.tsx          # BibTeX
        └── SidebarNav.tsx            # Sticky TOC with scroll-spy
```

## Data sources

All in `public/data/` (kept in sync with `../results/`):

| File | Contents |
|---|---|
| `layered_d20_r16_full.json` | Conditions A–F on Mini-Engram-d20, 20 users |
| `layered_rag_full.json` | RAG conditions G–J, 20 users |
| `layered_rag_scale_v2.json` | KB sweep N∈{34..1000}, Mini-Engram |
| `qwen_rag_full.json` | Qwen-3B + RAG, single KB (34) |
| `qwen_rag_scale_v2.json` | KB sweep N∈{34..1000}, Qwen-3B |
| `multihop_rag.json` | 8-pair chained-fact RAG |
| `latency_table.csv`, `rag_scale_table.csv` | Aggregate CSVs |

## Adding a new visualization

1. Add a loader to `src/lib/data.ts` (return a clean view-model, not raw JSON).
2. Write a component in `src/components/` that uses Recharts.
3. Add a section in `src/App.tsx` with an `id` matching the sidebar entry.

## Tech notes

- Tailwind v3 (not v4) — uses standard PostCSS pipeline.
- Recharts is the only chart lib (~110 KB gzipped). Build warns about a 600 KB
  bundle; fine for an academic site, can be code-split later if it matters.
- Sticky sidebar uses an IntersectionObserver scroll-spy (no router).
- Charts are pure data → recharts, no Plotly/D3 — keeps the bundle small.
