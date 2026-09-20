
## 1. Documentation Rules

- Write documentation as though you're sitting beside me narrating a story conveyed by the file. Assume the reader has read the README and whatever comes earlier in the
pipeline. Default to stating "this file/function does/returns X"; Use first person plural "we" when explaining an approach we take.

- Keep comments and docstrings as concise as possible.

- Never talk about diffs and what the code does not do as opposed to a previous
version, the documentation always reflects one snapshot of time: the current repository. 

- The docstring at the top of a file must state its purpose and what resides in it.

  - For files that transform something, state INPUT and OUTPUT at the top.
  - For files centered on one process, state PROCESS as numbered steps.
  - For files that are meant to be invoked, end the docstring with a `Run:` block holding the literal command, with the flags actually used. 

- Files with parts are split it into them with banner comments. Under each
banner, two or three lines say what happens in that section. 

- The thing that drives the file comes first, right after the docstring, and its
numbered steps name the whole story within one screen. The sections that follow
appear in the order those steps call them, so the file reads top to bottom.

- Chunk function bodies with blank lines, and a comment conveying what the next
chunk does, if and only if it is not immediately obvious what the code does. 

- A comment states what happens and carries the reader along the story. It should not 
debate constant values or argue for and against design choices by providing citations. 
If such information comes up, add it to design-choices.md.

- Name variables for maximal narrative clarity. 

## 2. Modularise but only if necessariy

- Values that always travel together belong in a dataclass.

- If a files revolves around only this dataclass, upgrade it to a Class with all
  functions used by it, as either static or normal methods.

- Keep signatures as narrow as its call sites need. 

## 3. Review as you write

- At the level of each chunk, each function, and the file, ask two things.

  - Can this be simpler? Name what breaks without it. "Nothing", "an error message gets less specific" and "a case that cannot arise" all mean delete it now. If it has to stay but reads as more than it is, rewrite it until it states its own intent.

  - Did anything force it to be this way — a constraint, a definition, an interface, a measurement? If not, you chose, and I am carrying a decision I do not know about. Write it into design-choices.md as you make it, not afterwards.
