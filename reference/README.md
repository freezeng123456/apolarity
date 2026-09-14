# Isolated reference implementations

`stde_upstream/` is a source snapshot of the public STDE repository used only
to verify equation names, case numbers, and model defaults when designing the
benchmark protocol.  It is deliberately outside `src/apolarity` and is not
imported by the benchmark runners.  The primary experiment compares only
`nested_ad` and `rank_optimal`.

The snapshot was copied from the local checkout used for the protocol audit;
the original repository metadata is not included in this source tree.  Keep
the accompanying provenance record with any published result so that changes
to the upstream reference cannot be confused with changes to the proposed
method.
