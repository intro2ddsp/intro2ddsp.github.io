# Audio Synthesis

Audio synthesis, the artificial production of sound, has been an active field of research for over a century. 
Early inventions included entirely new categories of musical instrument, such as Cahill's [Telharmonium](https://en.wikipedia.org/wiki/Telharmonium) and the [Theremin](https://en.wikipedia.org/wiki/Theremin), and the first systems for artificial speech production, such as Dudley's [Vocoder](https://ccrma.stanford.edu/~jos/pasp/Dudley_s_Vocoder.html).
The latter half of the 20th century saw a proliferation of research into digital methods for sound synthesis, built on advances in both digital signal processing and numerical methods.
Whilst once the purview of specialist research laboratories and pioneering music composers, applications of audio synthesis have since come to permeate daily life, from music, through synthetic voices, to the sound design in films, TV shows, video games, and even the cockpits of cars.

## Neural Audio Synthesis

In recent years, the field has undergone something of a technological revolution. The publication of WaveNet {cite}`oord_wavenet_2016`, an autoregressive neural network which produced a quantised audio signal sample-by-sample, first illustrated that deep learning might be a viable methodology for audio synthesis.
Over the following years, new methods for neural audio synthesis abounded, from refinements to WaveNet {cite}`oord_parallel_2018` to the application of entirely different classes of generative model, including generative adversarial networks (GANs), variational autoencoders, and more recently denoising diffusion models. 
Nonetheless, modelling audio signals remained challenging for deep neural networks.
Upsampling layers, crucial components of workhorse architectures such as GANs and autoencoders, were found to cause undesirable signal artifacts {cite}`pons_upsampling_2021`.
Similarly, frame-based estimation of audio signals was also found to be more challenging than might naïvely be assumed, due to the difficulty of ensuring phase coherence between successive frames, where frame lengths are independent of the frequencies contained in a signal {cite}`engel_gansynth_2019`.
