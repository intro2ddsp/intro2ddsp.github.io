# Differentiable Synthesis Libraries

There exist several open-source code implementations of differentiable DSP
components for creating and working with DDSP synthesizers. 

This includes implementations of the differentiable sinusoidal modelling synthesizer used in the work by Engel et al. and synthesis libraries modelled on modular synthesizers.

## DDSP TensorFlow

<a href="https://github.com/magenta/ddsp"><img src="https://img.shields.io/badge/Github-DDSP_TensorFlow-7DA416.svg?style=flat-square&amp;logo=GitHub" alt="Full repository"></a>

The official GitHub repository for Engel at al.'s DDSP.

## DDSP PyTorch

<a href="https://github.com/acids-ircam/ddsp_pytorch"><img src="https://img.shields.io/badge/Github-DDSP_PyTorch-7DA416.svg?style=flat-square&amp;logo=GitHub" alt="Full repository"></a>

Implementation of the DDSP model using PyTorch by ACIDS IRCAM.

## torchsynth

<a href="https://github.com/torchsynth/torchsynth"><img src="https://img.shields.io/badge/Github-torchsynth-7DA416.svg?style=flat-square&amp;logo=GitHub" alt="Full repository"></a>

A GPU-enabled modular synthesizer library written in PyTorch. Provides of a number of different synthesis modules that are inspired by classic modular synthesizers that can be combined to create larger synthesizer voices. Achieves over 16k times faster than realtime GPU performance for fast synthetic dataset generation. {cite:p}`turian_one_2021`

- [Documentation](https://torchsynth.readthedocs.io/en/latest/)
- [Example - GPU THX Sound](https://colab.research.google.com/drive/143JkEfo-TwNPvvMvhv7RGiLZpqzwVQYV?usp=sharing)

## SynthAX

<a href="https://github.com/PapayaResearch/synthax"><img src="https://img.shields.io/badge/Github-SynthAX-7DA416.svg?style=flat-square&amp;logo=GitHub" alt="Full repository"></a>

A modular synthesizer based on torchsynth built with JAX. Achieves over 80k times faster than realtime generation speeds. {cite:p}`cherep2023synthax`

## References

```{bibliography}
:filter: docname in docnames
```
