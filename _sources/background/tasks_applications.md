# Tasks and Applications

DDSP audio synthesis has found application in a diversity of domains and downstream
tasks. Most of the literature falls under the domain music, speech, or singing voice synthesis.
However, recent work has also explored sound effect synthesis {cite:p}`barahona2023noisebandnet`.
We note that while singing voice could be considered a subtask of music, there are significant
enough differences in implementation and connection to speech synthesis such that
we decided to include it as a separate domain.

In this section we'll go over the more common synthesis tasks related to DDSP audio synthesis, focusing
on providing an introduction to music and singing voice synthesis. For a more comprehensive account of tasks and applications, see
our recent review on the topic {cite:p}`hayes_review_2023`.

## Tasks Overview
Here is an overview of the most common synthesis tasks that DDSP has been used for. Related
tasks across domains are listed along the horizontal with a brief decsription of the task.

```{figure} ../images/tasks_applications.png
---
name: tasks_and_applications
---
Overview of tasks that DDSP audio synthesis has found application. Boxes with a dashed outline indicate that this tasks have yet to be explored within the DDSP literature, as far as we are aware.
```

In this tutorial we'll focus on music and singing domains, although we encourage interested
readers to explore the related speech synthesis literature. Many of the techniques used
in speech synthesis are applicable to musical audio (and certainly singing voice) synthesis,
and vice-versa.

## Musical Instrument Synthesis

The primary task in DDSP audio synthesis for musical audio synthesis has revolved around 
the modeling of instruments. The goal is to use a differentiable digital synthesizer to
accurately model the tones of a target instrument using a data-driven approach. For example,
the original DDSP work by {cite:t}`engel_ddsp_2020` implemented a sinusoidal modelling
synthesizer {cite:p}`serra_spectral_1990` differentiably and trained a neural network
to predict synthesizer parameters from audio features The input audio features included fundamental frequency,
loudness, and mel-frequency cepstral coefficents (MFCCs). 
They trained this neural network using recordings of instrumental performances, including violin performances.

Subsequent to the differentiable sinusoidal modelling synthesizer by Engel et al., a number
of other digital synthesis methods have been explored within a differentiable paradigm including
waveshaping synthesis {cite:p}`hayes_neural_2021`,
frequency modulation synthesis {cite:p}`caspe_ddx7_2022, ye_nas-fm_2023`, and wavetable 
synthesis {cite:p}`shan_differentiable_2022`. 
The design of the synthesizers mentioned thus far are particularly well-suited for modelling
of monophonic and harmonic instruments. This is in part due to the use of explicit
pitch detection and harmonically constrained differentiable synthesizers, which adds a
useful inductive bias towards the generation of these sounds.

**How about polyphonic or non-harmonic sounds?**

While these sounds are more challenging to model
for a number of reasons (which we'll start to investigate further in this tutorial), some
research has been conducted in this direction. {cite:t}`renault_differentiable_2022` explored polyphonic generation
for piano synthesis and {cite:t}`caillon_rave_2021` used a hybrid approach combining
neural audio synthesis with DDSP for polyphonic generation. Synthesis of non-harmonic
rigid-body percussion sounds was explored by {cite:t}`diaz_rigid-body_2022`.

Let's look at how differentiable musical instrument synthesizers have been applied to
other creative tasks.

### Timbre Transfer

Timbre transfer is related to the style transfer task in the image domain, which aims to
apply the style of one image or artist to a target image {cite:p}`gatys_neural_2015`. In 
musical timbre transfer the goal is to apply the timbre from one instrument to the performance
of another. For example, we might try to replicate a trumpet performance but have it sound
like it was played on a violin. This task is related to singing voice conversion, discussed
below.

The synthesis approach proposed by {cite:p}`engel_ddsp_2020` naturally lends itself to this
task. Fundamental frequency $f_0$ and loudness envelopes are explicitly presented to the model which
then predicts the timbre via time-varying harmonic amplitudes. Timbre is essentially
encoded in the weights of the neural network during training. A model trained on violin
performances can then be used for many-to-one timbre transfer by providing $f_0$ and loudness
envelopes from a new source instrument.

```{figure} ../images/tone_transfer.png
---
height: 250px
name: tone_transfer
---
Screen shot of the [tone transfer website](https://sites.research.google/tonetransfer). Accessed
November 1, 2023.
```

You can try this out for yourself online using [Google's ToneTransfer](https://sites.research.google/tonetransfer). Details on the development of this web app can be read in the paper by {cite:t}`carney_tone_2021`.

### Performance Rendering

The task of performance rendering extends a musical instrument synthesis through the
prediction of synthesis parameters from symbolic notation (e.g., MIDI). This involves
not only correctly representing musical attributes such as pitch, dynamics, and rhythm,
but also capturing the expressive performance elements. 
An example of a recent performance rendering system by {cite:t}`wu_midi-ddsp_2022` augmented Engel et al.'s differentiable sinusoidal modelling synthesizer with a neural
network front-end to predict synthesis parameters from MIDI.

### Sound Matching

Sound matching is the inverse problem of determining optimal
parameters for a synthesizer to match a target audio sample.
Prior to DDSP, solutions using neural networks were limited to
supervised training on a parameter loss (i.e., using a synthetic dataset where correct
synthesizer parameters are known). {cite:t}`masuda_improving_2023` proposed a differentiable
subtractive synthesizer for sound matching, allowing for an audio loss function to be
used and enabling training on **out-of-domain** sounds.

## Singing Voice Synthesis

Singing voice synthesis (SVS) in the context of DDSP is the task of generating realistic singing
audio from input audio features. It draws from both speech synthesis and musical instrument synthesis.
The synthesizer in SVS is often referred to as a vocoder, which is the term used within
the speech literatue for a synthesizer. The musical context of SVS imposes
further challenges including emphasis on pitch and rhythmic accuracy, more expressive
pitch and loudness contours, and a demand for higher resolution results (i.e., higher sampling rate).

One of the first DDSP vocoders designed for SVS was SawSing by {cite:t}`wu_ddsp-based_2022`,
who proposed a differentiable source-filter approach using a sawtooth waveform for the 
excitation signal. Subsequent work on SVS has also explored a differentiable source-filter
method and include {cite:t}`golf`, who used differentiable linear predictive coding (LPC)
and wavetables, and {cite:t}`nercessian_differentiable_2023`, who proposed a differentiable
WORLD vocoder.

### Singing Voice Conversion

Singing voice conversion (SVC) is the task of transorming a recording of one singer such that it
sounds like it was sung by a different target sing. It is related to both timbre transfer
of musical instruments and voice conversion in speech synthesis. In addition to maintaining
the intelligibility of the sung lyrics, SVC systems must contend with the dynamic pitch
contours and expressivity present in singing voices. In the DDSP literature this task was
explored by {cite:t}`nercessian_differentiable_2023`, who applied their differentiable
WORLD vocoder to the task of SVC using a decoder conditioned on a speaker-independent embedding.
Another notable example of DDSP-based SVC is in the results of the 2023 SVC Challenge {cite:t}`huang_singing_2023`,
which reported strong performance from DSPGan {cite:p}`song_dspgan_2023`, 
a hyrbid model that uses a DDSP vocoder
to generate input features for a generative adversarial vocoder.

## Summary

In this section we presented a brief overview of some of the main applications and
tasks that DDSP audio synthesis has been applied to, focusing on the domains of music
and singing. Each domain includes a main synthesis task which involves generating audio
from input acoustic features. Timbre transfer in music and singing voice conversion in
singing voice synthesis is the task of transforming a performance such that it sounds like
it was performed on another instrument or sung by another singer, respectively. In the music
domain we also introduced performance rendering and synthesizer sound matching.

## References

```{bibliography}
:filter: docname in docnames
```
