---
layout: post
title: "Variational Autoencoder Framework Derivation"
author: "Parth Shah"
tags: VAE Variational Inference DL AI generative-model
---


# The Problem
The problem setup is quite simple. Given data $$X$$ we want to learn a generative model that can reconstruct the data $$\hat{X}$$ Figure 1 shows the generative process.[1]
Here $$Z$$ is the latent representation learned by VAE[2] and $$X$$ is the
data of dimensions $$N$$. With the problem well defined now, we can start deriving the framework.[3]

# VAE Framework Derivation
From Figure 1, we can write the following:

$$ p(z,x) = p(z)p(x\vert z) $$

Inspired from the Autoencoders, the terms in the above equation can be imagined as follows: $$p(x\vert z)$$ can be imagined as the probabilistic decoder and $$p(z\vert x)$$ can be imagined as the probabilistic encoder.

From Bayes theorem

$$ p(z\vert x) = \frac{p(x\vert z)p(z)}{p(x)} $$

Using conditioning, we can rewrite this as

$$ p(z\vert x) = \frac{p(x \vert  z)p(z)}{\int p(x \vert  u)p(u) du} $$

Assume:

$$ p(Z) \sim N(0, I) $$

And, we can paramaterize $$ p(X\vert Z) $$ for $$f \in F$$

$$ p(X\vert Z) \sim N(f(Z), cI) $$

where $$F$$ is a family of functions and $$c>0$$. For now assume $$f$$ is known.

Even though we know $$p(x\vert z)$$ and $$p(z)$$, we can't estimate $$p(z\vert x)$$ from Conditioning Eqn. because of the intractability of the denominator. Therefore, we resort to variational inference.

Approximate $p(z \vert x)$ with $q_x(z)$ s.t.,

$$
    q_x(z) \sim N(g(x), h(x))
$$

where $g \in G$ and $h \in H$, $G$ and $H$ are families of functions.
We want to fine $(g,h) \in G \times H$ s.t, the approximation captures most information of $p(z \vert x)$. We use KL Divergence to measure the <b> distance </b> between $q_x(z)$ and $p(z \vert x)$.

i.e.,

$$
\begin{aligned}
    (g^{*}, h^{*}) &=\underset{(g,h) \in G \times H}{\mathrm{argmin}} D_\text{KL}( q_x(z) \ \vert\vert\ p(z\vertx) ) \\
    &= \underset{(g,h) \in G \times H}{\mathrm{argmin}} E[log\ q_x( z)] - E[log\ p( z\vertx)]  & (\because D_\text{KL}(p(x) \ \vert\vert\ q(x) = E[log\ p( x)] - E[log(q(x))]) \\
    &= \underset{(g,h) \in G \times H}{\mathrm{argmin}} E[log\ q_x( z)] - E[\frac{ log\ p(x\vertz) p(z) }{p(x)}] & (\because p(z\vertx) = \frac{p(x\vertz)p(z)}{p(x)})\\
    &= \underset{(g,h) \in G \times H}{\mathrm{argmin}} E[log\ q_x( z)] - E[\frac{log\ p(z,x)}{log\ p(x)}] & (From\  \eqref{eq:model}) \\
    &= \underset{(g,h) \in G \times H}{\mathrm{argmin}} E[log\ q_x( z)] - E[log\ p( z,x)] - E[log\ p( x )] & (\because E[A+B] = E[A] +E[B]) \\
    &= \underset{(g,h) \in G \times H}{\mathrm{argmin}} E[log\ q_x( z)] - E[log\ p( z,x)] - log\ p(x) & (\because E(c) = c)
\end{aligned}
$$

This is still intractable because of the presence of $p(x)$.
