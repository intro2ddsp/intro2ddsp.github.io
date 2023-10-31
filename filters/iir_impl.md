# Differentiable Implementation of IIR Filters

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

This method is pretty fast in general because PyTorch natively support FFT and only requires a few computational nodes to record this. Moreover, in many cases, we define our loss functions in frequency domain, which makes more sense to use the frequency sampling method without converting it back to the time domain. 

However, there are some drawbacks of this method. It equals to approximating the IIR filter with an FIR filter because of discrete time Fourier transform (DFT), which sampled $N$ frequencies $[e^0, e^{-j \frac{2 \pi}{N}}, \dots, e^{-j \frac{2 \pi (N-1)}{N}}]$. Sampling in the frequency domain is equivalent to truncating the impulse response in the time domain and making it periodic every $N$ samples. 
The filtering becomes a circular convolution of the truncated impulse response. Moreover, $N$ is usually set to some large numbers to have a good resolution in the frequency domain, which can increase the memory usage.

## Custom Backward Functions

From the above sections, you can see that the main problem of differentiable IIR is the **massive computational graph**. If we can find a way to compress the number of nodes, for example, using one node that represents the whole IIR computation, then the problem is solved. But there is no equivalent operation in PyTorch that can do this. We have to write our own differentiable IIR operation, and this requires us to write a custom backward function for it.

```{mermaid}
flowchart LR
	a(a)
	x(x)
	y(y)
	IIR[IIR Forward]
	IIR_back[IIR Backward]
	da(dL/da)
	dx(dL/dx)
	dy(dL/dy)

	subgraph forward
		direction TB
		a --> IIR
		x --> IIR
		IIR --> y --> L
	end
	subgraph backward
		direction BT
		dy --> IIR_back
		a2(a) --> IIR_back
		y2(y) --> IIR_back
		IIR_back --> dx
		IIR_back --> da
	end

	forward ==> backward
```

### Coefficient gradients $\frac{\partial \mathcal{L}}{\partial a_k}$

If we directly take the derivative of {eq}`iir` with respect to $a_k$, we will get

$$
\frac{\partial y[n]}{\partial a_k} =  -y[n-k] - \sum_{i=1}^M a_i \frac{\partial y[n-i]}{\partial a_i}.
$$ (iir_dyda)
It is filtering the negative, shifted output with the same set of coefficients! We can get the gradients for different $k$ in one pass because their differences are just time-shifted versions of each other.

Then, given a loss function $\mathcal{L}$ and the backpropagated output gradients $\frac{\partial \mathcal{L}}{\partial \mathbf{y}}$, we can get the gradients $\frac{\partial \mathcal{L}}{\partial a_k}$ using chain rule

$$
\begin{bmatrix}
\frac{\partial \mathcal{L}}{\partial a_1} \\
\vdots \\
\frac{\partial \mathcal{L}}{\partial a_M}
\end{bmatrix} = 
\begin{bmatrix}
\frac{\partial \mathbf{y}}{\partial a_1} \\
\vdots \\
\frac{\partial \mathbf{y}}{\partial a_M}
\end{bmatrix}
\frac{\partial \mathcal{L}}{\partial \mathbf{y}}.
$$ (iir_dLda)

### Input gradients $\frac{\partial \mathcal{L}}{\partial x[n]}$

Let us recall that the derivative of a linear matrix transform $\mathbf{y} = \mathbf{H}\mathbf{x}$ with respect to $\mathbf{x}$ is $\mathbf{H}^T$. We can represent IIR as a matrix transformation with $\mathbf{H}_a$ which has infinite dimensions. Of course, the $\mathbf{x}$ we use here is also infinite dimensional $[\cdots, 0, x[0], x[1], \dots]$.

$$
\mathbf{H}_a = 
\begin{bmatrix}
\cdots & 0 & 0 & 0 & \cdots & 0\\
\ddots & \vdots & \vdots & \vdots &  & \vdots \\
\cdots & 1 & 0 & 0 & \cdots & 0 \\
\cdots & h_1 & 1 & 0 & \cdots & 0 \\
\cdots & h_2 & h_1 & 1 & \cdots & 0 \\
 & \vdots & \vdots & \vdots & \ddots & \vdots \\
\end{bmatrix}
$$ (iir_Ha)

The gradients $\frac{\partial \mathcal{L}}{\partial \mathbf{x}}$ can be computed using chain rule

$$
\frac{\partial \mathcal{L}}{\partial \mathbf{x}} = 
\frac{\partial \mathbf{y}}{\partial \mathbf{x}} 
\frac{\partial \mathcal{L}}{\partial \mathbf{y}} 
=  
\mathbf{H}_a^T \frac{\partial \mathcal{L}}{\partial \mathbf{y}}.
$$ (iir_dLdx)

The transposition $\mathbf{H}_a^T$ means the same filter but its impulse response is reversed. Thus, {eq}`iir_dLdx` can be re-written as

$$
\frac{\partial \mathcal{L}}{\partial x[n]} = 
\frac{\partial \mathcal{L}}{\partial y[n]} - \sum_{k=1}^M a_k \frac{\partial \mathcal{L}}{\partial x[n+k]}.
$$ (iir_dLdxn)

Notice the sign difference of the indexing inside the summation of {eq}`iir_dLdxn`.

### The full backward function

Combining {eq}`iir_dyda`, {eq}`iir_dLda` and {eq}`iir_dLdxn`, we get the full backward function for IIR, which consists of two IIR filtering and one matrix multiplication. Now, you can use any IIR filter implementation that is fast enough to fill you needs, as long as you register the backward functions, which also consists of IIR filters! This is the approach that is implemented in [TorchAudio's lfilter](https://pytorch.org/audio/stable/generated/torchaudio.functional.lfilter.html#torchaudio.functional.lfilter) and is documented in the GOLF paper {cite}`golf`.


```{mermaid}
graph LR
	da(dL/da)
	dx(dL/dx)
	dy(dL/dy)
	a(a)
	y(y)
	iir1[IIR]:::op
	iir2[IIR]:::op
	matmul[MatMul]:::op
	classDef op fill:#f96

	dy --> flip[reverse] --> iir1 --> flip2[reverse] --> dx
	a --> iir2
	a --> iir1
	y --> minus[-1] --> iir2
	iir2 --> matmul
	dy --> matmul --> da
	
```

## Parameterisation

One more thing that needs to be taken care of is the parameterisation of the coefficients. IIR filters can be unstable (i.e. the outputs are unbounded) if the roots of the denominator polynomial are outside the unit circle. Special cares need to be taken especially when the filter coefficients are predicted by a neural network.

### Complex conjugate pairs

One simple way is to predict the complex root $re^{-j\theta}$ inside the unit circle ($|r| \leq 1$) and add its conjugate so the filter coefficients are real. The value bound can be enforced by using the sigmoid function on the neural network output. The transfer function of this second-order IIR filter is then

$$
\begin{split}
H(z) &= \frac{1}{(1 - re^{-j\theta} z^{-1})(1 - re^{j\theta} z^{-1})} \\
&= \frac{1}{1 - 2r \cos(\theta) z^{-1} + r^2 z^{-2}}.
\end{split}
$$ (iir_complex)

Higher order IIR filters can be obtained by concatenating more second-order IIR filters. 

### Coefficient parameterisation

The other way is to directly predict the second-order IIR filter coefficients $a_1, a_2$. It is well known tha a second-order IIR filter is stable if 

$$
|a_1| < 1 + a_2, \quad |a_2| < 1.
$$ (iir_stable)

These constraints can be added easily with the tanh function on the neural network output {cite}`nercessian2021lightweight`. This parameterisation covers a wider range of coefficients  because the roots are not restricted to be conjugate pairs.


### Reflection coefficients

The two methods we have discussed do not assume any order on applying second-order filters if we want to build a longer ($M > 2$) IIR filter. Any filtering order results in the same transfer function. If all the filter coefficients are predicted by the same neural network, this can be problematic because the neurons in the last layer are ordered but their target is permutation invariant. We introduce the third parmetersiation that use an intermediate representation called reflection coefficients which is ordered.

Reflection coefficients $p_k$ are widely used in speech processing. Sometimes it is also called PARCOR coefficients. It relates to the ratio of the areas of the $k$-th and $(k+1)$-th sections of the vocal tract tube model. The system is gauranteed to be stable if $|p_k| < 1$. The filter coefficient $a_k$ can be computed from the reflection coefficients using Levinson-Durbin recursion

$$
a_k^i = 
\begin{cases}
p_k, & k = i \\
a_{k}^{i-1} - p_k a_{i-k}^{i-1}, & k < i \\
\end{cases} \quad ,
$$ (iir_levinson)

where $i$ is the iteration number and the final coeffecients are $a_k^M$. They are very compact representation and can be quantised for transmission.


## Summary

In this chapter, we have presented two different ways to implement differentiable IIR filters which are fast to compute. We have also discussed three different parameterisations of the filter coefficients, which values are bounded to ensure stability and can be parameterised by neural networks. In the next chapter, we will demonstrate how to implement these methods in PyTorch.


## References

```{bibliography}
:filter: docname in docnames
```