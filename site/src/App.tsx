import { Nav } from "./components/layout/Nav";
import { Scene } from "./components/layout/Scene";
import { Hero } from "./components/scenes/Hero";
import { Resources } from "./components/scenes/Resources";

import { TwoJobs } from "./components/visuals/TwoJobs";
import { PriceLocator } from "./components/visuals/PriceLocator";
import { BrainSplit } from "./components/visuals/BrainSplit";
import { ContaminationSplit } from "./components/visuals/ContaminationSplit";
import { WriteAFact } from "./components/visuals/WriteAFact";
import { GlassBox } from "./components/visuals/GlassBox";
import { LayeredBars } from "./components/visuals/LayeredBars";
import { KBCrossover } from "./components/visuals/KBCrossover";
import { ServingFlow } from "./components/visuals/ServingFlow";

export default function App() {
  return (
    <div className="relative grain-overlay">
      <Nav />
      <main>
        <Hero />

        <Scene
          id="two-jobs"
          index="01"
          kicker="The problem"
          title={<>Personal memory is <span className="display-italic text-engram">two jobs</span>, not one.</>}
          lede={<>Recall the fact — and reason over it. The same store has to do both, for millions of users, without leaking one user's facts into another's.</>}
        >
          <TwoJobs />
        </Scene>

        <Scene
          id="thesis"
          index="02"
          kicker="The thesis"
          title={<>Every method pays. The question is <span className="display-italic text-engram">where</span>.</>}
          lede={<>Each way of storing a fact reaches near-perfect recall — and each pays for it in weaker reasoning. They differ only in where that price lands, and how fast it grows.</>}
          wide
        >
          <PriceLocator />
        </Scene>

        <Scene
          id="brain"
          index="03"
          kicker="First principle"
          title={<>Borrowed from the <span className="display-italic text-engram">brain</span>.</>}
          lede={<>The hippocampus writes a sparse, local trace; the neocortex holds the slow, shared skill. Keeping them apart is what lets a new fact land without overwriting how you think.</>}
        >
          <BrainSplit />
        </Scene>

        <Scene
          id="contamination"
          index="04"
          kicker="Why not a LoRA?"
          title={<>A LoRA can't keep a <span className="display-italic text-engram">secret</span>.</>}
          lede={<>To store one fact, a per-user LoRA bends a function the whole model shares — so it changes text that has nothing to do with the user. An addressed write does not.</>}
          wide
        >
          <ContaminationSplit />
        </Scene>

        <Scene
          id="method"
          index="05"
          kicker="The method"
          title={<>Remembering a fact is <span className="display-italic text-engram">writing a few rows</span>.</>}
          lede={<>A fact decomposes into where (the trigger's hash → a sparse set of addresses) and what (the value to write). Nothing else in the model moves.</>}
          wide
        >
          <WriteAFact />
        </Scene>

        <Scene
          id="glassbox"
          index="06"
          kicker="The mechanism"
          title={<>Open the <span className="display-italic text-engram">box</span>.</>}
          lede={<>Unlike a LoRA, every step of the write can be watched on the trained model — and each one is measured, not assumed.</>}
          wide
        >
          <GlassBox />
        </Scene>

        <Scene
          id="layered"
          index="07"
          kicker="The design"
          title={<>Content and skill, kept <span className="display-italic text-engram">apart</span>.</>}
          lede={<>Per-user facts in the memory table; one shared reasoning skill for everyone. Together they beat every all-in-one baseline on every measure.</>}
          wide
        >
          <LayeredBars />
        </Scene>

        <Scene
          id="deployment"
          index="08"
          kicker="When it wins"
          title={<>Who wins depends on <span className="display-italic text-engram">deployment</span>.</>}
          lede={<>A per-user table never grows with the population. Drag the knowledge base: retrieval decays as the pool grows, while the layered design holds flat — and overtakes it past ~100 facts.</>}
          wide
        >
          <KBCrossover />
        </Scene>

        <Scene
          id="serving"
          index="09"
          kicker="At scale"
          title={<>Millions of users, <span className="display-italic text-engram">one table</span>.</>}
          lede={<>A ~50-line server swaps a user's rows in, runs the forward pass, restores. Per-request work is independent of how many users share the machine.</>}
          wide
        >
          <ServingFlow />
        </Scene>

        <Resources />
      </main>
    </div>
  );
}
