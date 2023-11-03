---
jupytext:
  formats: md:myst
  text_representation:
    extension: .md
    format_name: myst
    format_version: 0.13
    jupytext_version: 1.11.5
kernelspec:
  display_name: Python 3
  language: python
  name: python3
---

# Digital Filter Modelling

Linear time-invariant (LTI) digital filters are the workhorses of audio signal processing.
They are an indispensable component in algorithms from audio coding to artificial reverberation and, of course, audio synthesisers of many kinds.

In this section, we assume familiarity with the basics of digital filters.
For example, terms like _impulse response_, _z-transform_, and _frequency response_ should be familiar to you.
We refer readers who would like to improve their knowledge on the subject to Julius O. Smith III's excellent online textbook {cite}`smith_filters_2007`. 

The section starts by introducing differentiable formulations of _finite impulse response_ (FIR) filters, and uses these to create a time-varying filtered noise synthesiser.
Then, we turn our attention to _infinite impulse response_ filters, discussing their advantages and drawbacks, and presenting two approaches to differentiable implementation. 
