# Design decisions
This file contains documentation detailing the design decisions made during the development of grf-py. Its purpose is to offer insight into the decision-making process for those wishing to understand or revisit these choices later on. It's important to note that documenting a decision here doesn't imply immediate full compliance of the entire API with that decision. Instead, it signifies a gradual transition towards aligning with it.

## NML naming of properties
### Pros
- Smooth transition between NML and grf-py.
### Cons
- NML does a lot of magic with properties. Grf-py properties are more low-level and sometimes work differently.
- Some NML properties are named poorly and their role is not clear for people unfamiliar with NML.
- "Class" is a reserved word in Python so it's inconvenient to have a property named like that.
### Conclusion
Use NML naming where advantageous but remain flexible without excessive reliance on it.
