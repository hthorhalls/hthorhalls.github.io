---
title:  "The intuition behind self-attention"
subtitle: We explore the inner workings of self-attention, the core mechanism of Transformers, from a conceptual point of view. 
og_image: /assets/img/embedding-infusion.png
keywords: Machine learning
layout: default
---

Let's start with a problem definition - given a sequence of words we'd like to train a neural net to predict the next word in a sequence. Consider the following sentence:

*The soccer player took a deep breath and walked out onto the*
{: style="text-align: center; font-size: 1.5em"}

Now how about: 

*The musician took a deep breath and walked out onto the*
{: style="text-align: center; font-size: 1.5em"}

Notice how similar these sentences are, they're almost exactly the same. But the fact that one concerns an athlete and the other a musician dramatically affects what word is likely to come next. The former sentence hints at words like *field* or *pitch* while the latter could be *stage*. There are some key observations here: 

- Some words have a much larger effect on our prediction than others 
- The words with large predictive power might be quite far away from the word we're trying to predict

One way to model this problem is to define a *window*, commonly referred to as **context**, that defines how far back our neural net looks in a piece of text to predict the next word. Then, given a specific word, we infuse that word with the meaning of all words that came before it in the context. 

{% include image.html url="/assets/img/embedding-infusion.svg" description="Each word has the change to contribute and alter the meaning of successive words"  width="80%"%}

Using the example in the above image, we are not only looking at the word "the" on its own when predicting the next word, but the word *the* that has **captured the meaning of its context**. 

Conceptually, we can think of it like this:

1. "the" on its own has contributed the information that the next word is describing a specific noun. 
2. We can improve our prediction by infusing the meaning of the word "kicked", hinting that we are most likely looking for a word describing a physical object. 
3. The extra information from the phrase "soccer player" narrows it down even further. 

Infusing all this knowledge into "the", predicting the next word becomes a much more tractable task. Doesn't that sound similar to how we, humans, approach language? At least, this is how language is modeled in the **transformer architecture** {% cite vaswani2023attentionneed %} and it has worked better than anyone could have ever imagined. Let's zoom in and see how this works in detail.  

In practice, most language models operate at the sub-word level, with sub-words referred to as tokens. We'll use the term **token** going forward but assume a 1:1 mapping between tokens and words for simplicity.

## Token embeddings

First, we need to convert our tokens to numbers that our neural net can operate on. We can do that by converting each token to a numerical vector. We refer to these vectors as **embeddings**, a vector in space that *embeds* the meaning of a particular token. Token embeddings that are close to each other in the vector space should also be close semantically and vice versa. 

{% include image.html url="/assets/img/embedding-explainable.svg" description="Tokens with similar meaning lie close together in the vector space"  width="40%"%}

We can think of each dimension in our embedding space as representing some sort of latent factor that encodes some information about our token. For example, one dimension could be encoding tone, another whether the token is a noun etc. 

In addition to the embeddings for each token we have **positional embeddings**. That is, we represent each position in our context with a numerical vector. The purpose of this is to give our neural net some information about the ordering of tokens. We simply add these embeddings together to get our final **static token embeddings**. 

{% katexmm %}
$$ x = x_{raw} + x_{pos} $$
{% endkatexmm %}

Using the geometrical interpretation of vector addition, we can think of the positional embeddings as a displacement for our raw token embeddings, the order of tokens alter their meaning slightly. But in what way? That's for our neural net to find out!
The raw token embeddings and the position embeddings are a subset of the trainable parameters of our neural net, they are learnt over the course of training. It turns out that tasks like next token prediction are quite good at producing these embeddings. 

But how can we make our embeddings communicate with each other? Remember, a token can influence the meaning of another token appearing much later in a sentence. Clearly embeddings provide some sort of basic summary information for a token on its own. How do we help our tokens communicate and change their meaning based on preceding tokens? 

## Contextual embeddings

This is where **attention** comes in. It is simply a mechanism for figuring out how the embedding of each token changes when it takes into account the tokens before it. The name attention comes from the idea that preceding tokens *attend* to successive tokens. Basically we want to compute **contextual embeddings** from our raw token embeddings. 

Consider the example: 

*The knight in shining*
{: style="text-align: center; font-size: 1.5em"}

{% katexmm %}
Let's denote the embedding vector for the word *shining* as $x$. Now the contextual embedding of shining can be expressed as: 


$$x_{contextual} = x + \Delta x $$

An example of a way to formulate this is to represent $\Delta x$ as some weighted sum of all preceding token embeddings. Sticking with our example, the embedding change for the token *shining* becomes: 

$$\Delta x_{shining} = w_1 x_{the} + w_2 x_{knight} + w_3 x_{in} + w_4 x_{shining} $$
{% endkatexmm %}

These weights are referred to as **attention scores**. 

## Queries and keys

How do we come up with these attention scores, how much should a token should contribute to the change in meaning of another token?  It turns out that we can formulate this problem as a *search problem*. That is, **for each token** we define two vectors: 
    
* **Query**: What we are looking for?
* **Key**: What do I broadcast to match relevant queries? 


{% katexmm %}

We can think of the **query** is a numerical vector that encodes a question - for example: *"Which tokens contribute a positive sentiment?"*. 

Just like with our embeddings, we want our neural net to figure out which question to ask, and the question might be slightly different based on what token is asking the question. We can express this mathematically as a linear transformation, a matrix $W_Q$ that projects a token embedding $x$ to a lower dimensional query vector $q$. Let's denote the lower dimension as $d$. Note that this dimension is typically much lower than the dimension of the embedding space since it's only encoding information pertaining to a small aspect of a token. 

We can think of the **key** vector as an answer that each token broadcasts to queries - *"Hey I'm a token with a pretty positive sentiment"*. 

Identically to the query we define a linear transformation $W_K$ that takes a token embedding $x$ and projects it a key vector $k$ with the same dimension $d$. 

{% include image.html url="/assets/img/q-k-space.svg" description="Example of a query and key vector in a d=2 space"  width="100%"%}

Given these vectors, how do we calculate our *attention score*, or how well each token answers each query? There is no shortage of similarity functions to choose from. However, transformers specifically use the **dot product** to compute this similarity. Let's denote how well a token $i$ attends to the token *shining* with $a_i$. We can now formulate how well the token *knight* attends to the token *shining*: 

$$ a_{knight} = q^T_{shining} k_{knight} = (W_q x_{shining})^T (W_k x_{knight}) $$

It turns out it's helpful to normalize our attention scores using [*softmax*](https://en.wikipedia.org/wiki/Softmax_function) so they represent a probability distribution, i.e. all the attention scores for a specific token query sum up to 1. This prevents attention scores from being too extreme. 

## Values

We can now apply our *attention scores* to the each embedding to calculate our contextual embedding change $ \Delta x $. However, rather than take a linear combination of the embeddings themselves, it is helpful to allow the network to learn what part of the embedding is relevant to the update. For a given query, with its value vector a token is communicating: *"If I'm relevant you should only tweak these parts of your embedding"*. 

We define our third linear transformation $W_v$ that takes an embedding $x$ and returns another embedding sized vector $v$ that represents the **value vector**. We now have everything we need to calculate the change in meaning for our shining token! 

$$\Delta x_{shining} = a_{the} v_{the} + a_{knight} v_{knight} + a_{in} v_{in} + a_{shining} v_{shining} $$

All these three vectors, *query*, *key* and *value* constitute a single **attention head**. 

## Multiple attention heads and deepening our network

We can see how a single attention head changes the meaning of our token given the context and a certain query. We can expand this and add several attention heads and aggregate their information. Sticking with our example: 


$$ E_{shining} = \Delta E_{head1}  + \Delta E_{head2} + \Delta E_{head3} + \ldots$$
{% endkatexmm %}

When training our neural net, our hope is that the heads stay diffuse and end up encoding a different set of queries. Maybe one is related to sentiment, one grammar etc. 
We then add a stack of feed forward layers, layers of neurons with non-linear activations to capture the relationship between the components in our new contextual embeddings. 

{% include image.html url="/assets/img/transformer-layer.svg" description="A simplified high level overview of a single 'transformer layer': Multi-head attention block + two feedforward layers"  width="40%"%}


However, this only constitutes a single transformer layer. The transformer architecture then stacks several of these layers together into a deep neural net. To get a feel for how big these neural nets can get, GPT-3 had 96 transformer layers {% cite brown2020languagemodelsfewshotlearners %}. Similar to how early layers in convolutional neural nets learn edges, the early attention heads encode some some low level syntactic contextual information in our embeddings that focus on words close to them in the context. Later attention blocks then learn to query for higher level concepts and look further back in the context. For example, deep layers have been proven to identify co-reference - words refererring to the same thing. {% cite peters2018dissectingcontextualwordembeddings %}.

It is magical, how well this architecture works, given enough data. With a bit of fine-tuning, the text generation from transformers blows earlier turing tests out of the water.


## Sources

{% katexmm %}
I want to highlight two resources that really helped solidify my understanding of self-attention. In a way, this blog post is a lossy summary of these videos after going through my own biological transformer.

- Andrej Karpathy's [Let's build GPT: from scratch, in code, spelled out](https://www.youtube.com/watch?v=kCc8FmEb1nY)
- 3Blue1Brown's [video](https://www.youtube.com/watch?v=eMlx5fFNoYc) on self-attention

### Nitty gritty details 

This blog post is only meant to give a conceptual overview of attention in transformers. For further understanding it would be beneficial to go on reading about

* How to parallelize training on a piece of context using *masked attention*. 
* Ways to make training more stable using *scaled attention*, layernorm, dropout etc. 
* How we transform a piece of text into tokens using *tokenization* 


All good material for another blog post! 



{% endkatexmm %}













