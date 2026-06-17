import { usePageProgress } from "../../lib/hooks";
import { PAPER_URL, CODE_URL } from "../../data/headline";

export function Nav() {
  const p = usePageProgress();
  return (
    <>
      {/* progress rail */}
      <div className="fixed top-0 left-0 right-0 z-50 h-[3px] bg-transparent">
        <div
          className="h-full bg-engram origin-left"
          style={{ transform: `scaleX(${p})` }}
        />
      </div>
      <header className="fixed top-0 left-0 right-0 z-40">
        <div className="flex items-center justify-between px-6 md:px-12 lg:px-20 py-4">
          <a href="#top" className="flex items-center gap-2.5 group">
            <span className="grid grid-cols-1 gap-[2px] w-4">
              <i className="block h-[2px] rounded-full bg-rule" />
              <i className="block h-[2px] rounded-full bg-engram" />
              <i className="block h-[2px] rounded-full bg-rule" />
              <i className="block h-[2px] rounded-full bg-engram" />
            </span>
            <span className="font-mono text-[0.72rem] uppercase tracking-[0.22em] text-ink-100 group-hover:text-engram transition-colors">
              User&nbsp;as&nbsp;Engram
            </span>
          </a>
          <nav className="flex items-center gap-5 font-mono text-[0.72rem] uppercase tracking-[0.18em]">
            <a className="text-ink-50 hover:text-engram transition-colors" href={CODE_URL}>
              Code
            </a>
            <a
              className="px-3 py-1.5 rounded-full bg-ink text-paper-50 hover:bg-engram transition-colors"
              href={PAPER_URL}
            >
              Paper ↗
            </a>
          </nav>
        </div>
      </header>
    </>
  );
}
