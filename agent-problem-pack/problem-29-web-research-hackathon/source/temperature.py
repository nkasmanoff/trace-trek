"""Unrelated utility module.

This file is deliberately unrelated to the task. It exists only so the
workspace contains some code, mirroring a real repository where the answer
to a user's question is NOT present in the local files. Reading this code
will not help answer the question in the task prompt.
"""


def celsius_to_fahrenheit(celsius):
    return celsius * 9 / 5 + 32


def fahrenheit_to_celsius(fahrenheit):
    return (fahrenheit - 32) * 5 / 9
