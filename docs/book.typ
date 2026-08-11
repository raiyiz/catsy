#import "@preview/physica:0.9.8": *

#set page(
  paper: "a4",
  margin: (x: 2.5cm, top: 3cm, bottom: 2.5cm),
  header: align(right, text(fill: rgb("#1a3a5f").lighten(20%), size: 9pt)[Technische Dokumentation | CV Quantum Toolkit]),
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
  lang: "de"
)

#show heading: set text(fill: rgb("#1a3a5f"), font: "Liberation Sans")
#show heading.where(level: 1): it => {
  v(1.5em, weak: true)
  it
  v(1em, weak: true)
  line(length: 100%, stroke: 0.5pt + rgb("#1a3a5f").lighten(50%))
  v(0.5em)
}

// Title Page
#align(center)[
  #v(2cm)
  #text(size: 26pt, weight: "bold", fill: rgb("#1a3a5f"))[Architektur & Mathematische Spezifikation] \
  #v(0.5em)
  #text(size: 14pt, style: "italic", fill: gray.darken(30%))[Dokumentation des Continuous-Variable Quantum Optik Toolkits]
  #v(2cm)
]

#grid(
  columns: (1fr, 1fr),
  gutter: 1cm,
  [
    *Entwickelt für:* \
    Core Software Engineering \
    Quantum Optics Simulation Unit
  ],
  align(right)[
    *Dokument-Version:* 1.0 (Iterativ) \
    *Mathematische Notation:* Typst + `physica` \
    *Datum:* August 2026
  ]
)

#v(2em)
#line(length: 100%, stroke: 1pt + rgb("#1a3a5f"))
#v(1.5em)

#include "chapter1.typ"
#include "chapter2.typ"
#include "chapter3.typ"
#include "chapter4.typ"
