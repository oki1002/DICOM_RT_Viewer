"""errors.py — Failures a registration can report to its caller."""


class RegistrationError(ValueError):
    """A registration could not be run on the input it was given.

    Raised for a region too small to register, a template with no contrast,
    a non-positive control-point spacing — conditions a host application is
    expected to surface to the user and let them correct, as opposed to the
    ``RuntimeError`` SimpleITK raises when an optimisation itself fails.
    """
