# Objectives and constraints

`Compliance` measures the work the loads do on the structure, `u^T K u`. Lower compliance means a stiffer structure under the given loads, so minimizing it is the default objective. Nothing about it is elasticity-specific: the same response reads thermal compliance off a conduction model, since it only needs `element_energies` from the physics.

Stiffness alone has a trivial optimum: fill the whole domain with material. `Volume` reads back the material fraction of a region (the design region by default), and comparing it to a target turns it into the resource budget the optimizer works within, so `Volume() <= 0.4` means: build the stiffest structure that fits in 40% of the domain.

Every constraint is normalized to `g <= 0`: satisfied means non-positive. `Volume() <= 0.4` builds a `Constraint` whose `g` is the response's value minus the bound, scaled by the bound, so `g` is zero at the target and grows positive as the design exceeds it. The `<=`/`>=` operators on `ResponseBase` do this normalization for any response, so a custom one gets the same convention for free.

Minimizing compliance at a fixed volume, rather than minimizing volume at a fixed compliance, is the standard formulation because it is the easier one to pose. Volume is dimensionless and known up front: `Volume() <= 0.4` means the same thing on any mesh. A compliance bound has to be chosen ahead of the run and is sensitive to mesh resolution and load magnitude, so the reverse formulation needs a target that's hard to pick well, for no different an optimum, since both formulations trace out the same trade-off between stiffness and material use.
