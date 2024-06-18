# papisqa: LLMs for Papis

This plugin for [papis](https://github.com/papis/papis) allows you to discuss your library using large language models (LLM). We integrate [paper-qa](https://github.com/whitead/paper-qa/) with papis.

## Installation

Install papis (if not already installed):

```
$ pipx install papis
```

Then inject `papisqa`:

```
$ pipx inject papis git+https://github.com/isaksamsten/papisqa
```

## Usage

This assumes you already have a papis library set up and an OpenAI api-key set `export OPENAI_API_KEY=sk...`.

```
$ papis ai index
```

It will scan all documents in the default library and index them using OpenAI's embedding model. We use the cheapest one.

Next we can ask questions about the library:

```
$ papis ai ask "What is the difference between multirocket and Rocket?"
```

Using my library we get the following response:

```
# Question
What is the difference between multirocket and Rocket?

# Answer
MultiRocket expands upon Rocket by incorporating several key enhancements.
Unlike Rocket, which applies global max pooling and 'proportion of positive
values' (PPV) pooling, MultiRocket introduces three additional pooling
operations (Dempster et al., 2023 pages 6-7; Tan et al., 2022 pages 2-3).
MultiRocket transforms both the original time series and their first-order
difference, capturing rate of change (Tan et al., 2022 pages 7-7). It utilizes
the same 84 fixed kernels as MiniRocket but applies them to both the original
and differenced time series, resulting in broader and more nuanced feature
extraction (Tan et al., 2022 pages 7-8). These enhancements allow MultiRocket
to achieve higher accuracy with significantly improved speed, positioning it as
a state-of-the-art time series classification algorithm (Dempster et al., 2023
pages 6-7).

# References
 - Dempster et al., 2023 pages 6-7
 - Tan et al., 2022 pages 2-3
 - Tan et al., 2022 pages 7-8
 - Tan et al., 2022 pages 7-7
 - Tan et al., 2022 pages 6-7
```
