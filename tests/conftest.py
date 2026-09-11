"""Headless plotting for the whole suite. Any figure a test produces is
inspected as an object, never displayed, so a GUI backend can never block."""
import matplotlib

matplotlib.use("Agg")
