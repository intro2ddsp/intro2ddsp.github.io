# What is DDSP?

Aiming to address the challenges posed by neural audio synthesis, one line of research explored the integration of domain knowledge from speech and musical instrument acoustics, and digital signal processing, into neural networks.
This was achieved by implementing the building blocks of signal processing algorithms using an automatic differentiation framework such as [TensorFlow](https://www.tensorflow.org/), [PyTorch](https://pytorch.org/), or [Jax](https://github.com/google/jax).
These implementations could then be included in the forward pass of a neural network.

For example, a neural network might output a value which is used as the cutoff frequency of a filter, which is implemented differentiably.
During training, a loss function is computed on the output of the filter and, using the backpropogation algorithm, its gradient with respect to the neural network's parameters is computed. 
In order to perform this computation, the derivative of the filter's output with respect to its cutoff frequency must be evaluated.
That is to say, the filter forms a part of the computation graph, and its gradient is a factor of the chain rule decomposition of the loss gradient.

Whilst the DDSP approach has been applied for numerous reasons, a common motivation is to gain a domain-appropriate inductive bias by constraining a network's output space to that of a known signal model.
In this way, 

These methods included 
Whilst some methods combined the outputs of classical techniques with neural networks~\citep{valin_lpcnet_2019}, others integrated them by expressing the signal processing elements differentiably~\citep{wang_neural_2019-2,juvela_gelp_2019}. 
This was crystalised in the work of \cite{engel_ddsp_2020}, who introduced the terminology \textit{differentiable digital signal processing} (DDSP).
In particular, Engel et al. suggested that some difficulties in neural audio synthesis could be explained by certain biases induced by the underlying models.
Implementing the signal model differentiably allowed loss gradients to be backpropagated through its parameters, in a manner similar to differentiable rendering~\citep{kato_differentiable_2020}. 

In subsequent years, DDSP was applied to tasks including music performance synthesis~\citep{wu_midi-ddsp_2022,jonason_control-synthesis_2020}, instrument modelling~\citep{renault_differentiable_2022}, synthesiser sound matching~\citep{masuda_synthesizer_2021}, speech synthesis and voice transformation~\citep{choi_nansy_2022}, singing voice synthesis and conversion~\citep{yu_singing_2023,nercessian_differentiable_2023}, sound-effect generation~\citep{barahona-rios_noisebandnet_2023,hagiwara_modeling_2022}, and more.
The technology has also been deployed in a number of publicly available software instruments and real-time tools.\footnote{These include Google Magenta's DDSP-VST (\url{https://magenta.tensorflow.org/ddsp-vst}), Bytedance's Mawf (\url{https://mawf.io/}), Neutone Inc.'s Neutone (\url{https://neutone.space/}), ACIDS-IRCAM's \textit{ddsp\~} (\url{https://github.com/acids-ircam/ddsp_pytorch}) and Aalborg University's JUCE implementation (\url{https://github.com/SMC704/juce-ddsp}). Accessed 21st August 2023.}
Fig. \ref{fig:ddsp-overview} illustrates the general structure of a typical DDSP synthesis system and we list included papers in Table~\ref{tab:papers}.

Differentiable signal processing has also been applied in tasks related to audio engineering, such as audio effect modelling~\citep{carson_differentiable_2023,kuznetsov_differentiable_2020,lee_differentiable_2022}, automatic mixing and intelligent music production~\citep{steinmetz_style_2022,martinez_ramirez_differentiable_2021}, and filter design~\citep{colonel_direct_2022-1}.
Whilst many innovations from this work have found use in synthesis, and vice versa, we do not set out to comprehensively review these tasks areas.
Instead, we address this work where it is pertinent to our discussion of differentiable audio synthesis, and refer readers to the works of \citet{ramirez_deep_2020}, \citet{moffat_approaches_2019}, \citet{de_man_intelligent_2019}, for reviews of the relevant background, and to the work of \citet{steinmetz_deep_2022} for a summary of the state of differentiable signal processing in this field.

In light of the growing body of literature on DDSP, we hope for this article to benefit two groups.
The first consists of those unfamiliar with the details of DDSP, but who wish to gain an overview of the field.
We hope that some may even discover applications of these techniques in their work.
Indeed, we note that DDSP-based audio synthesis has already found use in more diverse tasks, including pitch estimation~\citep{engel_self-supervised_2020}, source separation~\citep{schulze-forster_unsupervised_2023}, and articulatory parameter estimation~\citep{sudholt_vocal_2023}.
Through discussion of the current understanding of the limitations of DDSP, we also endeavour to allow this group to better assess the suitability of these techniques for their work.
The second group consists of those already working with DDSP, who wish to gain a broader awareness of the field.
For this group, we endeavour in particular to provide a streamlined account of relevant technical contributions, and to pair this with a discussion of valuable future research directions.

The terms \textit{differentiable digital signal processing} and \textit{DDSP} have been ascribed various meanings in the literature.
For the sake of clarity, whilst also wishing to acknowledge the contributions of \cite{engel_ddsp_2020}, we therefore adopt the following disambiguation in this article:

\begin{enumerate}
    \item We use the general term \textit{differentiable digital signal processing} and the acronym \textit{DDSP} to describe the technique of implementing digitial signal processing operations using automatic differentiation software.
    \item To refer to Engel, et al.'s Python library, we use the term \textit{the DDSP library}.
    \item We refer to the differentiable spectral modelling synthesiser and neural network controller introduced by \cite{engel_ddsp_2020}, like other work, in terms of their specific contributions, e.g. \textit{Engel, et al.'s differentiable spectral-modelling synthesiser}.
\end{enumerate}
