# BBVS Video Understanding

BBVS turns time-aligned audiovisual evidence into structured knowledge notes. Its language separates stable machine decisions from reader-facing descriptions of the source content.

## Knowledge Structure

**Knowledge Item**:
An atomic, high-recall candidate extracted from source evidence, with its own content and source location. It is input to editorial selection, not automatically a report entry.
_Avoid_: Time segment, chapter

**Report Topic**:
A reader-level synthesis selected for the final report. It aggregates up to a few core Knowledge Items across Evidence Units and may attach a small number of supporting items.
_Avoid_: Sentence summary, Content Map entry, time slice

**Control Type**:
A stable English enum that classifies a Knowledge Item for machine filtering, validation, and report organization. Its vocabulary is intentionally small and domain-independent.
_Avoid_: Domain profile, free-form type

**Content Label**:
A concise Chinese description generated from the source that names a Knowledge Item in terms natural to its subject, such as `定理`, `反例`, or `设计原则`. It is reader-facing vocabulary and does not control program behavior.
_Avoid_: Control Type, hard-coded domain subtype

**Knowledge Relation**:
A domain-independent semantic link between Knowledge Items, such as support, dependency, instantiation, qualification, or refutation.
_Avoid_: Chronological adjacency

**Evidence Unit**:
A time-aligned envelope of speech and visual evidence used to locate and recover source material. It is evidence for Knowledge Items, not a writing chapter or a semantic category.
_Avoid_: Chapter, Knowledge Item

**Content Map**:
A high-recall inventory of candidate Knowledge Items. Its item count measures extraction coverage, not the number of knowledge points displayed in the report.
_Avoid_: Report outline, table of contents
