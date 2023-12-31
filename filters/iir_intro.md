# Infinite Impulse Response (IIR) Systems

A system with [infinite impulse response](https://en.wikipedia.org/wiki/Infinite_impulse_response) (IIR) is called an IIR system. It is described by the following difference equation

$$
\begin{split}
y[n] 
&= x[n] - a_1 y[n-1] - a_2 y[n-2] - \cdots - a_M y[n-M] \\
&= x[n] - \sum_{k=1}^{M} a_k y[n-k]
\end{split}
$$ (iir)

and its transfer function

$$
H(z) = \frac{1}{1 + \sum_{k=1}^{M} a_k z^{-k}},
$$ (iir_tf)

where $a_1, \dots, a_M$ are the system coefficients. 
Each output sample $y[n]$ is a weighted sum of the past output samples $y[n-1], \dots, y[n-M]$ and the current input sample $x[n]$. It is also called recursive system because the output samples are recursively computed from past output samples.
Conventionally, an IIR filter also depends on the past input samples $x[n-1], \dots, x[n-M]$ with another set of coefficients $b_0, b_1, \dots, b_M$ in the difference equation

$$
H(z) = \frac{b_0 + b_1 z^{-1} + \cdots + b_M z^{-M}}{1 + a_1 z^{-1} + \cdots + a_N z^{-M}}
$$

but here we restrict the discussion to systems that only depend on the past output samples for simplicity. 
This is called an all-pole system because it only has poles and no zeros in the transfer function. (Poles and zeros are the roots of the numerator and denominator of the transfer function, respectively.)

If we convert the equation {eq}`iir` to only have input samples on the right-hand side, we obtain

$$
\begin{split}
y[n] &= x[n] + \sum_{k=1}^{\infty} h_k x[n-k],\\
h_k &= \sum_{i=1}^k \sum_{q \in Q_k^i} \prod_{j=1}^i -a_{q_j}, \\
Q_k^i &= \{q \in \mathbb{Z}_{>0}^i \mid \sum_{j=1}^i q_j = k\}.
\end{split}
$$ (iir2)

The last two equations are ugly but thankfully we will not need them later in this notebook. They are only included for completeness. You can see that the output samples $y[n]$ are a weighted sum of the current and past input samples $x[n], x[n-1], \dots$, just like in the FIR case. However, the length of the impulse response is infinite, hence the name IIR.

IIRs are very useful because you can approximate long FIRs with much less number of coefficients in an IIR. This also implies that they are cheaper to compute. 

## Connection to Linear Predictive Coding (LPC)

In the speech literature, there is a similar system to IIR that has been used for speech coding and synthesis. It assumes that, the current speech sample $s[n]$ can be predicted from the past $M$ samples $s[n-1], \dots, s[n-M]$ with a linear combination of them. 

$$
\hat{s}[n] = \sum_{k=1}^{M} a_k s[n-k]
$$ (lpc)

This is called linear predictive coding (LPC). The prediction error is thus equals

$$
e[n] = s[n] - \hat{s}[n] = s[n] - \sum_{k=1}^{M} a_k s[n-k].
$$ (lpc_error)

If we switch the sides of the equation, we obtain the equation of an IIR system

$$
s[n] = e[n] + \sum_{k=1}^{M} a_k s[n-k].
$$ (lpc_iir)

Besides the minus/plus sign difference, it is exactly the same as the IIR equation in Equation {eq}`iir` with $x[n] = e[n]$ and $y[n] = s[n]$! The transfer function of this system also has physical meaning to the speech signal. It approximates the vocal tract transfer function if the vocal tract is modelled by a tube with $M$ sections. The coefficients $a_1, \dots, a_M$ are called the LPC coefficients and are compact representation for compression. We can call $e[n]$ as the _source_ and the transfer function as the _filter_, and together they form a source-filter model for speech. The _source_ usually means the signal that is generated by the vocal folds.

```{mermaid}
flowchart LR
	start(( ))
	stop(( ))
	filt[Vocal Tract FIlter]

	start -- source --> filt -- speech --> stop
```

## Connection to Recurrent Neural Networks (RNN)

The recurrent structure of IIR systems is also very similar to recurrent neural networks (RNN). In RNNs, there is a hidden state $\mathbf{h}[n]$ that is updated at each time step $n$ with the current input $\mathbf{x}[n]$ and the previous hidden state $\mathbf{h}[n-1]$.

$$
\begin{split}
\mathbf{h}[n] = f(\mathbf{A} \mathbf{x}[n] + \mathbf{B} \mathbf{h}[n-1]),
\end{split}
$$ (rnn)
where $f$ is a non-linear activation function, $\mathbf{A}$ and $\mathbf{B}$ are weight matrices.

You can see that the RNN equation can equals to a first-order IIR when $f(x) = x, \mathbf{A} = 1, \mathbf{B} = -a_1$, with scalar inputs and outputs. This similarity also explains why training IIRs in deep learning frameworks like PyTorch are slow. By flatting out the recurrent computational graph, we obtain a very deep network with number of layers equals to the number of time steps. The slow training problem is worse on IIRs because to have enough time context for audio applications, more than ten thousand of audio samples are needed, while RNNs usually trained on sequences with length of hundreds. We will revisit this problem in the later chapters when we talk about differentiable implementation of IIRs.

```{mermaid}
flowchart LR
	h1(h):::hidden
	h2(h):::hidden
	h3(h):::hidden
	hN(h):::hidden
	x1[x]:::input
	x2[x]:::input
	x3[x]:::input
	xN[x]:::input
	classDef hidden fill:#f96
	classDef input fill:#0098ba

	x1 -- A --> h1 -- B --> h2 -- B --> h3 -- ... --> hN
	x2 -- A --> h2
	x3 -- A --> h3
	xN -- A --> hN
```

## Summary

By the end of this chapter, you should know that
- IIR systems can be used to approximate long FIRs with much less number of coefficients.
- IIR systems are similar to linear predictive coding (LPC) and recurrent neural networks (RNN).

In the next chapter we are going to review some implementations for training IIRs in deep learning frameworks.

## References

```{bibliography}
:filter: docname in docnames
```