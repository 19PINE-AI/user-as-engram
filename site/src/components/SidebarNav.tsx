import { useEffect, useState } from "react";

interface Section { id: string; title: string; }

export function SidebarNav({ sections }: { sections: Section[] }) {
  const [active, setActive] = useState<string>(sections[0]?.id ?? "");

  useEffect(() => {
    const obs = new IntersectionObserver(
      (entries) => {
        for (const e of entries) {
          if (e.isIntersecting) setActive(e.target.id);
        }
      },
      { rootMargin: "-30% 0px -60% 0px", threshold: 0 },
    );
    sections.forEach(s => {
      const el = document.getElementById(s.id);
      if (el) obs.observe(el);
    });
    return () => obs.disconnect();
  }, [sections]);

  return (
    <nav className="sticky top-8 text-sm">
      <div className="text-[10px] font-mono uppercase tracking-widest text-ink-500 mb-2">
        Contents
      </div>
      <ul className="space-y-1 border-l border-ink-200">
        {sections.map(s => (
          <li key={s.id}>
            <a href={`#${s.id}`}
               className={`block -ml-px pl-3 py-1 border-l-2 transition-colors no-underline ${
                 active === s.id
                   ? "border-accent-600 text-accent-700 font-medium"
                   : "border-transparent text-ink-600 hover:text-ink-900 hover:border-ink-400"
               }`}>
              {s.title}
            </a>
          </li>
        ))}
      </ul>
    </nav>
  );
}
