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

# Physical Modeling and DDSP

In modeling acoustic systems, our primary focus is on two intricately linked differential equations: Newton's second law of motion and the Wave Equation.

These equations, in various forms and modifications, serve as the cornerstone for modeling a wide range of acoustic phenomena. Armed with these foundational equations, we encounter two principal challenges for sound synthesis and modeling:

- Numerical Integration: Our first challenge revolves around the numerical integration of these equations, enabling us to conduct physical simulations with a given set of initial parameters. This process forms the bedrock of simulating acoustic systems accurately.

- System Identification: Our second challenge centers on identifying the parameters within these equations when armed with observations from either simulated scenarios or real-world phenomena. This step is crucial for fine-tuning our models to match the complexities of real-world acoustic systems.

In this tutorial, we will explore two different methods that tackle these issues with the help of some classical formulations and gradient descent.
