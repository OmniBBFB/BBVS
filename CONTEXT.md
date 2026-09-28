# BBVS Video Understanding

BBVS turns time-aligned audiovisual evidence into structured knowledge notes. Its language separates stable machine decisions from reader-facing descriptions of the source content.

## Knowledge Structure

**Knowledge Item**:
An atomic piece of meaning extracted from source evidence, with its own content, source location, and role in the resulting notes.
_Avoid_: Time segment, chapter

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
