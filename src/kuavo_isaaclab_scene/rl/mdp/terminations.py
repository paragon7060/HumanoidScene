"""Keep termination predicates separate from shaping rewards."""

from .commands import task


def success(env):
    return task(env).success


def unsafe(env):
    return task(env).failure
