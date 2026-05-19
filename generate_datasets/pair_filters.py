"""Shared filters for final paired lie/truth datasets."""


def normalize_response(text):
    return " ".join(str(text or "").split()).strip()


def responses_equal(a, b):
    return normalize_response(a) == normalize_response(b)
