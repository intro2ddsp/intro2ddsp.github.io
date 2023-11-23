# Introduction

Welcome to the online supplement to the tutorial on "*Introduction to Differentiable Audio Synthesiser Programming*", presented at the [24th International Society for Music Information Retrieval Conference](https://ismir2023.ismir.net/) at [Politecnico di Milano](https://www.polimi.it/), November 5th --- 9th 2023.
This web book contains all content presented at the tutorial, including code examples and references. It also contains further material, going into greater depth on the topics covered.


## Motivation & Aims

The field of differentiable digital signal processing (DDSP) emerged with the incorporation of components, such as linear synthesis filters {cite}`juvela_gelp_2019` and spectral modelling synthesisers {cite}`engel_ddsp_2020`, into the end-to-end training of neural networks for speech and musical instrument synthesis.
It has since grown to encompass applications including pitch estimation {cite}`engel_self-supervised_2020`, source separation {cite}`schulze-forster_unsupervised_2023`, physical parameter estimation {cite}`sudholt_vocal_2023`, synthesiser sound matching {cite}`masuda_improving_2023`, and beyond.
By introducing a strong inductive bias, DDSP methods can often lead to a reduction in model complexity and data requirements, and provide an intermediate representation that is inherently interpretable in terms of familiar parametric signal processors.

Yet despite the growing popularity of such methods in research, the implementation of differentiable audio synthesizers is not always clearly documented, and the simple formulation of many synthesizers can obscure what often turns out to be complex optimization behaviour.
This tutorial aims to address this gap through an introduction to the fundamentals of differentiable synthesizer programming.
In particular, we hope that researchers in adjacent fields may find applications for these techniques in their work.

### Who is this for?

This book and the accompanying tutorial are aimed at music and audio researchers and engineers who wish to gain a practical understanding of how differentiable digital signal processors can be implemented and used in audio synthesis.
The content is targeted at those with a grounding in the fundamentals of signal processing and machine learning, although references to educational resources are provided where relevant for those who wish to refresh or supplement their existing knowledge.
We assume prior knowledge of Python 3.
The tutorial content is written using the [PyTorch](https://pytorch.org/) machine learning framework.
We do not assume prior experience with PyTorch, and attempt to provide sufficient background to allow the tutorial content to be understood.
Nonetheless, we encourage those who are totally new to PyTorch to explore the [official tutorials](https://pytorch.org/tutorials/beginner/basics/intro.html) to provide further context.

### A note on scope: we're *not* training neural networks

The majority of applications of DDSP involve its use in combination with a neural network, which typically produces the parameters of the signal processing operation (e.g. the cutoff frequency of a filter, or harmonic amplitudes).
In this tutorial and web book, however, we focus only on the components of this system _after_ the parameters are given --- that is, we concern ourselves only with the differentiable signal processing.
Nonetheless, all of the techniques we present can be (and mostly have been) composed with a neural network, as a direct consequence of their implementation using differentiable operations in a machine learning framework.

<!-- Computational research in music, speech, and other kinds of audio has undergone something of a revolution in recent years. -->
<!-- Deep learning has emerged as not only a viable and frequently superior option for many tasks --- from source separation, to music tagging, to music synthesis --- but has also allowed entirely new applications to be explored.  -->
<!-- Throughout this period of research, audio synthesis proved to be a particularly challenging task for deep neural networks, due to its high dimensionality, multi-scale temporal dependencies, and the frequently non-intuitive relationships between perception and signal characteristics. -->
<!-- These obstacles prompted numerous innovations, including WaveNet {cite}`oord_wavenet_2016` and multi-resolution adversarial objectives {cite}`kong_hifi-gan_2020`, which enabled rapid progress, particularly in the field of speech synthesis. -->
<!-- In parallel, the field of differentiable digital signal processing (DDSP) began to emerge, with the incorporation of a differentiable linear synthesis filter {cite}`juvela_gelp_2019` and spectral modelling synthesiser {cite}`engel_ddsp_2020` into end-to-end neural network training.  -->


<!-- Whilst many of the most exciting developments are predicated on the growing feasibility of scale in certain domains -- *more* data, *more* computation, *more* parameters -- these variables are not always under the researcher's control. -->
<!-- Frequently, tasks in music research suffer from a paucity of appropriate data, and on-demand access to large amounts of GPU computation is not always economically feasible or justifiable. -->

<!---->
<!-- ```{tableofcontents} -->
<!-- ``` -->

## Getting Started

This web book consists of a series of Jupyter notebooks, which can be explored statically on this page. To run the notebooks yourself, you will need to clone the Git repository:

```bash
git clone https://github.com/intro2ddsp/intro2ddsp.github.io.git && cd intro2ddsp.github.io
```

Then, you should create a Python virtual environment, using `virtualenv`, `conda`, or similar:

```bash
python -m venv venv && source venv/bin/activate
```

or:

```bash
conda create --name intro2ddsp python==3.10 && conda activate intro2ddsp
```

Next, you should install the dependencies:

```bash
pip install -r requirements.txt
```

And finally, you can launch the Jupyter notebook server:

```bash
jupyter notebook
```

## About the Authors

[*Ben Hayes*](https://benhayes.net/) is a final year PhD student at the Centre for Digital Music’s CDT in Artificial Intelligence and Music, Queen Mary University of London. His research focuses on expanding the capabilities of differentiable digital signal processing by resolving optimisation pathologies caused by symmetry. His work has been accepted to leading conferences in the field, including ISMIR, ICLR, ICASSP, ICA, and the AES Convention, and published in the Journal of the Audio Engineering Society. He has worked as a Research intern at Sony Computer Science Laboratories in Paris and ByteDance's Speech Audio and Music Intelligence team in London. He was also Music Lead at the award-winning generative music startup Jukedeck, and an internationally touring musician signed to R&S Records.

[*Jordie Shier*](https://jordieshier.com/) is a first year PhD student in the Artificial Intelligence and Music (AIM) programme based at Queen Mary University of London (QMUL), studying under the supervision of Prof. Andrew McPherson and Dr. Charalampos Saitis. His research is focused on the development of novel methods for synthesizing audio and the creation of new interaction paradigms for music synthesizers. His current PhD project is on real-time timbral mapping for synthesized percussive performance and is being conducted in collaboration with Ableton. He was a co-organizer of the 2021 Holistic Evaluation of Audio Representations (HEAR) NeurIPS challenge and his work has been published in PMLR, DAFx, and the JAES. Previously, he completed an MSc in Computer Science and Music under the supervision of Prof. George Tzanetakis and Assoc. Prof. Kirk McNally.

[*Chin-Yun Yu*](https://yoyololicon.github.io/) is a first year PhD student in the Artificial Intelligence and Music (AIM) programme based at Queen Mary University of London (QMUL), under the supervision of Dr György Fazekas. His current research theme is on leveraging signal processing and deep generative models for controllable, expressive vocal synthesis. In addition, he is dedicated to open science and reproducible research by developing open-source packages and contributing to public research projects. He received a BSc in Computer Science from National Chiao Tung University in 2018 and was a research assistant at the Institute of Information Science, Academia Sinica, supervised by Prof. Li Su. His recent work has been published at ICASSP.

[*David Südholt*](https://dsuedholt.github.io/publications/) is a first year PhD student in the Artificial Intelligence and Music (AIM) programme based at Queen Mary University of London (QMUL). Supervised by Prof. Joshua Reiss, he is researching parameter estimation for physical modelling synthesis, focussing on the synthesis and expressive transformation of the human voice. He received an MSc degree in Sound and Music Computing from Aalborg University Copenhagen in 2022, where he was supervised by Prof. Stefania Serafin and Assoc. Prof. Cumhur Erkut. His work has been published at the SMC conference and in the IEEE/ACM Transactions on Audio, Speech and Language Processing.

[*Rodrigo Diaz*](http://www.eecs.qmul.ac.uk/people/profiles/diazfernandezrodrigomauricio.html) is a PhD candidate in Artificial Intelligence and Music at Queen Mary University in London, under the supervision of Prof. Mark Sandler and Dr. Charalampos Saitis. Rodrigo’s work has been published in leading computer vision and audio conferences, including CVPR, ICASSP, IC3D, and the AES Conference on Headphone Technology. Before starting his PhD studies, he worked as a researcher at the Immersive Communications group at the Fraunhofer HHI Institute in Berlin, where he investigated volumetric reconstruction from images using neural networks. His current research focuses on real-time audio synthesis using neural networks for 3D objects and drums. Rodrigo’s interdisciplinary background includes a Master’s degree in Media Arts and Design from Bauhaus University in Weimar and a Bachelor of Music from Texas Christian University.

