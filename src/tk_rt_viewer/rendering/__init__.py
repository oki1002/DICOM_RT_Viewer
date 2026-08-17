"""Rendering sub-package: canvas-rendering collaborators for DicomViewer.

An explicit ``__init__.py`` (rather than an implicit namespace package) is
required for PEP 561 to apply: the ``py.typed`` marker in the top-level
package only covers *regular* sub-packages, so without this file type
checkers silently skip the inline annotations in ``rendering.*``.

Nothing is re-exported here on purpose; import the concrete modules
(``tk_rt_viewer.rendering.isodose``, ...) or the top-level package.
"""
