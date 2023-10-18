# Infinite Impulse Response (IIR) Systems

A system with infinite impulse response (IIR) is called an IIR system {cite}`dafx`. It is described by the following difference equation

$$
\begin{split}
y[n] 
&= x[n] - a_1 y[n-1] - a_2 y[n-2] - \cdots - a_M y[n-M] \\
&= x[n] - \sum_{k=1}^{M} a_k y[n-k]
\end{split}
$$ (eq:iir)

and the transfer function

$$
H(z) = \frac{1}{1 + \sum_{k=1}^{M} a_k z^{-k}},
$$ (eq:iir_tf)

where $a_1, \dots, a_M$ are the system coefficients. Each output sample $y[n]$ is a weighted sum of the past output samples $y[n-1], \dots, y[n-M]$ and the current input sample $x[n]$. It is also called recursive system because the output samples are recursively computed from past output samples, or all-pole system because the denominator of the transfer function is a polynomial of all poles.

If we convert the equation to only have input samples on the right-hand side, we obtain

$$
\begin{split}
y[n] &= x[n] + \sum_{k=1}^{\infty} h_k x[n-k],\\
h_k &= \sum_{i=1}^k \sum_{q \in Q_k^i} \prod_{j=1}^i -a_{q_j}, \\
Q_k^i &= \{q \in \mathbb{Z}_{>0}^i \mid \sum_{j=1}^i q_j = k\}.
\end{split}
$$ (eq:iir2)

The last two equations are ugly but thankfully we will not need them later in this notebook. They are only included for completeness. You can see that the output samples $y[n]$ are a weighted sum of the current and past input samples $x[n], x[n-1], \dots$, just like in the FIR case. However, the length of the impulse response is infinite, hence the name IIR.

IIRs are very useful because you can approximate long FIRs with much less number of coefficients in an IIR. This also implies that they are cheaper to compute.

## Connection to Linear Prediction (LP)

## Connection to Recurrent Neural Networks (RNN)

## Differentiable Implmentation

### Frequency Sampling Method

### Custom Backward Functions