---
layout: post
title: "Variational Autoencoder Framework Derivation"
author: "Parth Shah"
tags: [VAE, Variational, Inference, DL, AI]
---


# The Problem
The problem setup is quite simple. Given data $X$ we want to learn a generative model that can reconstruct the data $\hat{X}$ Figure 1 shows the generative process.[1]
Here $Z$ is the latent representation learned by VAE[2] and $X$ is the
data of dimensions $N$. With the problem well defined now, we can start deriving the framework.[3]

# VAE Framework Derivation
From Figure 1, we can write the following:
$$ p(z,x) = p(z)p(x|z) $$

Inspired from the Autoencoders, the terms in the above equation can be imagined as follows: $p(x\|z)$ can be imagined as the probabilistic decoder \& $p(z\|x)$ can be imagined as the probabilistic encoder.

From Bayes theorem

$$ p(z|x) = \frac{p(x|z)p(z)}{p(x)} $$

Using conditioning, we can rewrite this as

$$ p(z\|x) = \frac{p(x\|z)p(z)}{\int p(x\|u)p(u) du} $$

Assume:

$$ p(Z) \sim N(0, I) $$

And, we can paramaterize $p(X\|Z)$ for $f \in F$

$$ p(X|Z) \sim N(f(Z), cI) $$

where $F$ is a family of functions and $c>0$. For now assume $f$ is known.

Even though we know $p(x\|z)$ \& $p(z)$, we can't estimate $p(z\|x)$ from Conditioning Eqn. because of the intractability of the denominator. Therefore, we resort to variational inference.
