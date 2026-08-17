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
  #text(size: 14pt, style: "italic", fill: gray.darken(30%))[Documentation of the Continuous-Variable Quantum Optics Toolkit]
  #v(2cm)
]

#grid(
  columns: (1fr, 1fr),
  gutter: 1cm,
  [
    *Developed for:* \
    Core Software Engineering \
    Quantum Optics Simulation Unit
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
