# Differentiable Implmentation of IIR Filters

IIR is a linear system so it is differentiable everywhere. However, directly implementating {eq}`iir` is not a good idea because it is really slow in practice. The reason is because in deep learning framework, each arithmetic operation register a node in the computation graph, which will be used to backpropagate the gradients. For IIR, the number of nodes equals to the number of time steps, which is usually in the order of more than ten thousand. Creating these nodes results in significant overhead. In this chapter, we will discuss ways to mitigate this problem.

```{note}
FIR filters does not require such massive computational graph because its output samples are independent of each other, thus can be parallelised and merge into a single node. 
```


## Frequency Sampling Method

An alternative method that has been widely accepted is to do the computation in the frequency domain. Remember we have the transfer function $H(z)$ {eq}`iir_tf`. We can evaluate the transfer function on the unit circle $z = e^{j \omega}$ to obtain the frequency response $H(e^{j \omega})$ of the IIR filter

$$
\begin{split}
H(e^{j \omega}) 
&= \frac{1}{1 + \sum_{k=1}^M a_k e^{-j k \omega}} \\
&= \frac{1}{
    \begin{bmatrix}
    1 & a_1 & \cdots & a_M
    \end{bmatrix}
    \begin{bmatrix}
    e^0 \\
    e^{-j \omega} \\
    \vdots \\
    e^{-j M \omega}
    \end{bmatrix}
},
\end{split}
$$ (iir_fft)
which can be computed easily using the fast Fourier transform (FFT). The filtering process is then equivalent to multiplying the frequency response with the input signal $X(e^{j \omega})H(e^{j \omega}) $ in the frequency domain and then inverse FFT back to the time domain. 

This method is pretty fast in general because PyTorch natively support FFT and only requires a few computational node to record this. Moreover, in many cases, we define our loss functions in frequency domain, which makes more sense to use the frequency sampling method without converting it back to the time domain. 

However, there are some drawbacks of this method. 


## Custom Backward Functions

## Parameterisation