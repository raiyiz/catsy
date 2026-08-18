#import "@preview/physica:0.9.8": *

#set page(
  paper: "a4",
  margin: (x: 2.5cm, top: 3cm, bottom: 2.5cm),
  header: align(right, text(fill: rgb("#1a3a5f").lighten(20%), size: 9pt)[Technical Documentation | CV Quantum Toolkit]),
  footer: [
    #set text(fill: gray, size: 9pt)
    #grid(
      columns: (1fr, 1fr),
      [Continuous-Variable Quantum Optics Framework],
      // align(right, counter(page).display())
    )
  ]
)

#set text(
  font: "Liberation Serif",
  size: 11pt,
  lang: "en",
)

#show heading: set text(
  fill: rgb("#1a3a5f"),
  font: "Liberation Sans",
)

#show heading.where(level: 1): it => {
  v(1.5em, weak: true)
  it
  v(1em, weak: true)
  line(length: 100%, stroke: 0.5pt + rgb("#1a3a5f").lighten(50%))
  v(0.5em)
}
#show heading.where(level: 2): set text(size: 14pt)
#show heading.where(level: 3): set text(size: 12pt, style: "italic")

// -- Code styling ------------------------------------------------------
// Fenced code blocks: GitHub-flavored panel with a colored accent bar,
// bled slightly past the body-text margins for extra breathing room on
// long lines, and set in a narrower monospace so 90+ char lines still fit.
#show raw.where(block: true): it => pad(x: -0.6cm)[
  #block(
    fill: rgb("#f6f8fa"),
    stroke: (left: 2.5pt + rgb("#1a3a5f"), rest: 0.6pt + rgb("#d0d7de")),
    radius: 4pt,
    inset: (x: 10pt, y: 9pt),
    width: 100%,
    above: 1.2em,
    below: 1.3em,
    breakable: true,
    text(font: "DejaVu Sans Mono", size: 8.3pt, it)
  )
]

// Inline code: a soft pill so identifiers stand out from prose without
// competing with the fenced blocks above.
#show raw.where(block: false): it => box(
  fill: rgb("#eef1f5"),
  outset: (y: 2.6pt),
  inset: (x: 3pt),
  radius: 2pt,
  text(font: "DejaVu Sans Mono", size: 9.3pt, fill: rgb("#9c2f6b"), it)
)

// Title Page
#align(center)[
  #v(2cm)
  #text(size: 26pt, weight: "bold", fill: rgb("#1a3a5f"))[Architecture & Mathematical Specification] \
  #v(0.5em)
  #text(size: 14pt, style: "italic", fill: gray.darken(30%))[:: pHoCk aRouNd anD fiNd ouT ::]
  #v(2cm)
]

#grid(
  columns: (1fr, 1fr),
  gutter: 1cm,
  [
    *Developed for:* \
    tHe Inksty toot phore Narmelite
  ],
  align(right)[
    *Document version:* 1.1 (Iterative) \
    *Mathematical notation:* Typst + `physica` \
    *Date:* August 2026
  ]
)

#v(2em)
#line(length: 100%, stroke: 1pt + rgb("#1a3a5f"))
#v(1.5em)

#include "chapter1.typ"
#include "chapter2.typ"
#include "chapter3.typ"
#include "chapter4.typ"
#include "chapter5.typ"
#include "chapter6.typ"
#include "chapter7.typ"
#include "chapter8.typ"
#include "chapter9.typ"
#include "chapter10.typ"


= Core scientific literature
The documentation uses the following references as its principal scientific background. Chapter-specific literature is listed at the end of the relevant chapter; the list below is intended as a compact starting point for readers who want a systematic treatment.

- #link("https://doi.org/10.1103/RevModPhys.84.621")[C. Weedbrook et al., “Gaussian quantum information,” *Reviews of Modern Physics* 84, 621–669 (2012).]
- #link("https://doi.org/10.1103/RevModPhys.77.513")[S. L. Braunstein and P. van Loock, “Quantum information with continuous variables,” *Reviews of Modern Physics* 77, 513–577 (2005).]
- #link("https://www.routledge.com/Quantum-Continuous-Variables-A-Primer-of-Theoretical-Methods/Serafini/p/book/9781032157238")[A. Serafini, *Quantum Continuous Variables: A Primer of Theoretical Methods*, 2nd ed. (CRC Press, 2024).]
- #link("https://doi.org/10.1140/epjst/e2012-01532-4")[S. Olivares, “Quantum optics in the phase space: A tutorial on Gaussian states,” *EPJ Special Topics* 203, 3–24 (2012).]
- #link("https://doi.org/10.1002/3527602976.ch4")[W. P. Schleich, *Quantum Optics in Phase Space* (Wiley-VCH, 2001), especially the chapters on Wigner functions and quantum states in phase space.]
- #link("https://doi.org/10.1103/PhysRevLett.84.2722")[L.-M. Duan, G. Giedke, J. I. Cirac, and P. Zoller, “Inseparability criterion for continuous variable systems,” *Physical Review Letters* 84, 2722–2725 (2000).]
- #link("https://doi.org/10.1103/PhysRevLett.84.2726")[R. Simon, “Peres-Horodecki separability criterion for continuous variable systems,” *Physical Review Letters* 84, 2726–2729 (2000).]
- #link("https://doi.org/10.1103/PhysRevA.61.032302")[T. Opatrný, G. Kurizki, and D.-G. Welsch, “Improvement on teleportation of continuous variables by photon subtraction via conditional measurement,” *Physical Review A* 61, 032302 (2000).]
- #link("https://doi.org/10.1038/nature11902")[G. Kirchmair et al., “Observation of quantum state collapse and revival due to the single-photon Kerr effect,” *Nature* 495, 205–209 (2013).]
- #link("https://doi.org/10.1038/sdata.2016.18")[M. D. Wilkinson et al. et al., “The FAIR Guiding Principles for scientific data management and stewardship,” *Scientific Data* 3, 160018 (2016).]
- #link("https://doi.org/10.1371/journal.pcbi.1005510")[G. Wilson et al., “Good enough practices in scientific computing,” *PLoS Computational Biology* 13, e1005510 (2017).]
