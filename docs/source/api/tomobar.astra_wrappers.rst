:orphan:

ASTRA projector package moved
=============================

The historical Python package ``tomobar.astra_wrappers`` has been removed.
See :doc:`tomobar.projectors._astra` for the current implementation layout.

Use :class:`tomobar.reconstruction.direct.RecToolsDIR`,
:class:`tomobar.reconstruction.iterative.RecToolsIR` and
:class:`tomobar.projectors.operator.ProjectionOperator` directly.

Implementation import migration:

* ``astra_base`` is now ``tomobar.projectors._astra.base``.
* ``astra_tools2d`` is now ``tomobar.projectors._astra.parallel2d``.
* ``astra_tools3d`` is now ``tomobar.projectors._astra.parallel3d``.
* ``geometry_adapter`` is now ``tomobar.projectors._astra.geometry``.

These are private implementation paths; prefer the public operator API for
external integrations. This page preserves the historical documentation URL.
