# The Wave Equation

In the previous notebook, we explored the dynamics of a system of interconnected undamped oscillators in 1D. Each of these oscillators was connected to its neighbors by springs. When considering an individual simple harmonic oscillator, the restoring force was directly proportional to the displacement of the oscillator from its equilibrium position. However, in the context of our chain of oscillators, the restoring force becomes proportional to the difference in displacement between neighboring oscillators. Thus, for a single oscillator in the chain, the equation becomes:

$$
m \ddot{x} = -k \Delta x,
$$

where $\Delta x$ represents the difference in displacement between the oscillator and its neighbors.
As the spacing between oscillators approaches zero, this difference approaches the second spatial derivative of $x$, leading us to the wave equation:

$$
\ddot{x} = c^2 \frac{\partial^2 x}{\partial x^2}
$$

where $c^2 = \frac{k}{m}$ is the wave speed.

So far, we have only seen one method of solving the equation, namely using modal analysis and gradient descent. However, several methods exists. In paricular we will focus in the next section the solution to the wave equation using a network of waveguides which is inspired by the [travelling wave solution](https://en.wikipedia.org/wiki/D%27Alembert%27s_formula) of D'Alembert, where the solution is given by waves travelling on opposite directions:

$$
u(x, t) = F(x + ct) + G(x - ct)
$$

here $F(x + ct)$ represents a wave traveling to the left and $G(x - ct)$ represents a wave traveling to the right.
