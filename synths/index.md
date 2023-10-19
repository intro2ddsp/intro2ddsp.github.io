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

# Digital Synthesizer Modelling

In this section we'll cover some of the basics of implementing a synthesizer 
differentiably using PyTorch.

During the first few sections we'll work on building up the core components of a
sinusoidal modelling synthesizer, which is the same one that Engel et al. used to
model instrument tones in [paper]. Our goal will be to develop a synthesizer and use
gradient descent optimization to reproduce a tone from a musical instrument.

Can our synthesizer recreate these tones?

[insert some sounds]

The core idea of sinusoidal modelling synthesis was introduced by Serra and Smith in
the early 1990s. It builds on Fourier's theorem which states that any sound can be
decomposed into a set of complex sinusoidal functions. Sinusoidal modelling synthesis (SMS)
uses this concept within an analysis-synthesis pipeline for the resynthesis of 
instrumental tones. 

In theory, with enough sinusoids, any sound can be reproduced.
However, in practice we are interested in modelling sounds that include stochastic 
elements that are not well modelled by sinusoids and would require a large number of them
for accurate representation. Serra and Smith proposed to include filtered white noise
to handle these sonic components and this method was adopted in the Engel et al.'s
differentiable sinusoidal modelling synthesizer.

In this chapter we will focus on building up the harmonic synthesizer that Engel et al
used.

Sinusoidal modelling is also just one of many synthesis methods that we can model 
differentiably. Some of this alternative methods will be overviewed and we will also
introduce the torchsynth library which has implemented some of the core signal processing
objects into a differentiable modular synthesizer.
