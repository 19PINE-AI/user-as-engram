"""Figure 1: the layered design as a single, truly-layered schematic.

Top layer  -- the Engram memory table drawn as one long strip of slots:
mostly grey (general knowledge written at pretraining), a few coloured
slots per user (their personal facts), scattered by trigger-N-gram hash
address so users never overlap.
Bottom layer -- the frozen Mini-Engram backbone carrying one shared LoRA
(the reasoning skill), drawn full-width to signal "shared by everyone above".

This replaces the old three-panel (a)/(b)/(c) version: one image, one idea.
The six-condition Pareto lives separately in fig_pareto_layered.pdf.
"""
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Rectangle, FancyArrowPatch
from pathlib import Path

OUT = Path("/home/ubuntu/user-as-engram/paper/figs")
OUT.mkdir(parents=True, exist_ok=True)

# Colour-blind-friendly palette (ColorBrewer Dark2 for users; green for skill)
C_GREY   = "#d9d9d9"   # pretrained general knowledge
C_GREY_E = "#9a9a9a"
C_A      = "#1f77b4"   # User A
C_B      = "#d95f02"   # User B
C_C      = "#7570b3"   # User C
C_SKILL  = "#117733"   # shared LoRA
C_TEXT   = "#222222"

plt.rcParams.update({
    "font.family": "serif",
    "font.size": 12,
    "axes.titlesize": 14,
    "legend.fontsize": 11,
})

fig = plt.figure(figsize=(11.0, 4.3))
ax = fig.add_axes([0.0, 0.0, 1.0, 1.0])
ax.set_xlim(0, 112); ax.set_ylim(0, 43); ax.axis("off")

# ---------------------------------------------------------------------------
# TOP LAYER: the Engram memory table as one long strip of slots
# ---------------------------------------------------------------------------
N = 40
x0, pitch, slot_w = 4.0, 2.55, 2.15
y_tab, h_tab = 28.5, 6.2

users = {C_A: {3, 21, 31}, C_B: {9, 27}, C_C: {15, 37}}
slot_owner = {}
for col, idxs in users.items():
    for i in idxs:
        slot_owner[i] = col

for i in range(N):
    col = slot_owner.get(i, C_GREY)
    edge = C_GREY_E if i not in slot_owner else "#333333"
    lw = 0.6 if i not in slot_owner else 1.1
    x = x0 + i * pitch
    ax.add_patch(Rectangle((x, y_tab), slot_w, h_tab,
                           facecolor=col, edgecolor=edge, lw=lw,
                           alpha=0.95 if i in slot_owner else 1.0))
# "continues" marker
xend = x0 + N * pitch
ax.text(xend + 1.0, y_tab + h_tab / 2, r"$\cdots$", fontsize=17,
        va="center", color="#555")
ax.text(xend + 4.6, y_tab + h_tab / 2, "millions\nof rows", fontsize=10,
        va="center", ha="left", color="#777", style="italic")

# Top-layer heading
ax.text(x0, y_tab + h_tab + 3.4, "1.  Content",
        fontsize=15.5, color=C_TEXT, fontweight="bold", va="bottom")
ax.text(x0 + 14.5, y_tab + h_tab + 3.4,
        "— each user's facts as local Engram-row overrides",
        fontsize=12, color="#444", va="bottom")
ax.text(x0, y_tab + h_tab + 1.0,
        "Engram memory table  (addressed by trigger N-gram)",
        fontsize=11, color="#666", va="bottom", style="italic")

# Locality callout pinned to a right-side User-A slot (clears the heading)
xa = x0 + 31 * pitch + slot_w / 2
ax.annotate("write a user's fact = override only its rows\n"
            r"($\Delta$bpb on all other text $= +0.0001$)",
            xy=(xa, y_tab + h_tab), xytext=(xa - 4, y_tab + h_tab + 8.0),
            fontsize=11, color=C_A, ha="center",
            arrowprops=dict(arrowstyle="->", color=C_A, lw=1.2))

# ---------------------------------------------------------------------------
# Legend row
# ---------------------------------------------------------------------------
leg_y = 23.2
items = [(C_GREY, C_GREY_E, "pretrained general knowledge"),
         (C_A, "#333", "User A facts"),
         (C_B, "#333", "User B facts"),
         (C_C, "#333", "User C facts")]
lx = x0
for fc, ec, label in items:
    ax.add_patch(Rectangle((lx, leg_y - 0.2), 2.2, 2.2, facecolor=fc,
                           edgecolor=ec, lw=0.8))
    ax.text(lx + 2.9, leg_y + 0.9, label, fontsize=11, va="center",
            color=C_TEXT)
    lx += 5.0 + len(label) * 1.42
ax.text(xend + 4.6, leg_y + 0.9, "distinct addresses\n⇒ users never overlap",
        fontsize=10, va="center", ha="left", color="#777", style="italic")

# ---------------------------------------------------------------------------
# BOTTOM LAYER: frozen backbone + one shared LoRA, full width
# ---------------------------------------------------------------------------
y_base, h_base = 5.5, 13.0
base = FancyBboxPatch((x0, y_base), xend - x0 + 6.5, h_base,
                      boxstyle="round,pad=0.2,rounding_size=0.4",
                      linewidth=1.3, edgecolor="#555", facecolor="#f4f4f4")
ax.add_patch(base)

lora = FancyBboxPatch((x0 + 4, y_base + 4.3), xend - x0 - 1.5, 5.0,
                      boxstyle="round,pad=0.2,rounding_size=0.3",
                      linewidth=1.4, edgecolor=C_SKILL, facecolor=C_SKILL,
                      alpha=0.16)
ax.add_patch(lora)

cx = (x0 + xend) / 2
ax.text(x0 + 1.2, y_base + h_base - 1.6, "frozen Mini-Engram backbone",
        fontsize=10, color="#777", va="center", style="italic")
ax.text(cx, y_base + 7.0, "2.  Reasoning skill",
        fontsize=15.5, color=C_SKILL, ha="center", fontweight="bold")
ax.text(cx, y_base + 4.4,
        "one shared LoRA  —  trained once across other users, amortized over everyone",
        fontsize=12, color="#225522", ha="center")

# Spanning bracket beneath the LoRA to stress "shared by all users above"
ax.annotate("", xy=(x0 + 5, y_base + 2.2), xytext=(xend, y_base + 2.2),
            arrowprops=dict(arrowstyle="<->", color="#999", lw=1.0))
ax.text(cx, y_base + 1.0, "the same skill serves every user above",
        fontsize=10, color="#888", ha="center", style="italic")

# ---------------------------------------------------------------------------
# Connector between the two layers (content sits on the shared substrate)
# ---------------------------------------------------------------------------
for fx in (x0 + 6, cx, xend - 4):
    ax.add_patch(FancyArrowPatch((fx, y_tab - 0.3), (fx, y_base + h_base + 0.3),
                                 arrowstyle="-", color="#bbb", lw=0.8,
                                 linestyle=(0, (2, 2))))

fig.savefig(OUT / "fig_layered_arch.pdf", bbox_inches="tight", pad_inches=0.02)
print(f"wrote {OUT / 'fig_layered_arch.pdf'}")
plt.close(fig)
