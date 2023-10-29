# Finite Impulse Response

A digital filter is said to have finite impulse response (FIR) if its impulse response has finite support. This is equivalent to the condition that its difference equation depends only on samples from the input signal. In other words: it is not recursive.
This property guarantees numerical stability for a well-behaved impulse response. 
This is particularly desirable from the perspective of DDSP, where the exploding gradients of an unstable filter could destroy an entire training run. Further, FIR filters can be designed with desirable phase characteristics: linear phase in the causal case, and zero phase when non-causality is possible.

However, these properties come at the expense of greater computational and memory cost. Each sample in the filter's impulse response must be directly stored, and requires a multiply-and-add operation. 

```{figure} ../images/fir.png
---
height: 150px
name: fir-diag
---
A block diagram depicting the general causal finite impulse response digital filter. Reproduced from {cite}`smith_filters_2007`.
```

In the next section, we will present differentiable approaches to both applying an FIR filter to a signal, and designing an FIR filter with a desired frequency response. We will then combine these techniques to optimise a time-varying FIR filter to match an audio signal.
