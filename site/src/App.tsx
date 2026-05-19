import { Hero } from "./components/Hero";
import { Abstract } from "./components/Abstract";
import { KeyStats } from "./components/KeyStats";
import { Architecture } from "./components/Architecture";
import { ParetoSection } from "./components/ParetoSection";
import { LayeredHeadline } from "./components/LayeredHeadline";
import { KBScaleSection } from "./components/KBScaleSection";
import { ComparisonTableSection } from "./components/ComparisonTableSection";
import { MultihopSection } from "./components/MultihopSection";
import { DataBrowser } from "./components/DataBrowser";
import { CitationCard } from "./components/CitationCard";
import { SidebarNav } from "./components/SidebarNav";

const sections = [
  { id: "abstract", title: "Abstract" },
  { id: "stats", title: "Key Findings" },
  { id: "method", title: "Architecture" },
  { id: "layered", title: "Layered (A–F)" },
  { id: "rag", title: "RAG Comparison" },
  { id: "scale", title: "KB-Scale" },
  { id: "pareto", title: "Pareto" },
  { id: "multihop", title: "Multi-hop" },
  { id: "compare", title: "All Methods" },
  { id: "data", title: "Data Browser" },
  { id: "cite", title: "Cite" },
];

export default function App() {
  return (
    <div className="min-h-screen bg-white">
      <Hero />
      <div className="container-page flex gap-10 py-12">
        <aside className="hidden lg:block w-56 shrink-0">
          <SidebarNav sections={sections} />
        </aside>
        <main className="flex-1 min-w-0 space-y-4">
          <section id="abstract"><Abstract /></section>
          <section id="stats"><KeyStats /></section>
          <section id="method"><Architecture /></section>
          <section id="layered"><LayeredHeadline /></section>
          <section id="rag"><ComparisonTableSection focus="rag" /></section>
          <section id="scale"><KBScaleSection /></section>
          <section id="pareto"><ParetoSection /></section>
          <section id="multihop"><MultihopSection /></section>
          <section id="compare"><ComparisonTableSection focus="all" /></section>
          <section id="data"><DataBrowser /></section>
          <section id="cite"><CitationCard /></section>
        </main>
      </div>
      <footer className="border-t border-ink-200 mt-16">
        <div className="container-page py-10 text-sm text-ink-500 flex flex-col md:flex-row md:items-center justify-between gap-3">
          <div>© 2026 Bojie Li / Pine AI · User as Engram</div>
          <div className="flex gap-5">
            <a href="https://github.com/bojieli/user-as-engram">GitHub</a>
            <a href="#cite">BibTeX</a>
          </div>
        </div>
      </footer>
    </div>
  );
}
