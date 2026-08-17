"""State sub-package: Tkinter-independent observable state and caches.

An explicit ``__init__.py`` (rather than an implicit namespace package) is
required for PEP 561 to apply: the ``py.typed`` marker in the top-level
package only covers *regular* sub-packages, so without this file type
checkers silently skip the inline annotations in ``state.*``.

Nothing is re-exported here on purpose; import the concrete modules
(``tk_rt_viewer.state.viewer_state``, ...) or the top-level package.
"""
