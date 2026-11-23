# PolicyWinds labeler

The labeling engine behind [PolicyWinds](https://policywinds.com), which tracks U.S. state and federal bills on
firearms, transgender issues, abortion, immigration enforcement, and child custody and support.

For each bill a language model reads, the labeler returns which topics the bill really touches and, for each one,
**which way the bill moves that topic's fixed measure**: `expands`, `restricts`, `mixed` or `neutral`. Every
answer comes with a short quote from the bill, and the quote is checked against the bill's text.

Labels are produced by AI. The site's owner reviews labels, and a review always overrides the model; labels
that have not been reviewed are marked as AI labels on the site.

## The measures

Each topic has one fixed measure, worded so that "expands" and "restricts" do not take a side:

| Topic | "Expands" and "restricts" refer to |
|---|---|
| firearms | the ability to buy, own, carry or make firearms and accessories |
| transgender | legal recognition of, and access to services and facilities for, transgender people |
| abortion | legal access to abortion |
| immigration-enforcement | the scope of immigration enforcement |
| custody-support | the rights of the parent who pays support or has less custody, against the obligations placed on that parent |

A bill that requires more background checks therefore *restricts* firearms; a bill that bans sanctuary policies
*expands* immigration enforcement. The full definitions, subtags and scope rules are in
`src/policywinds_labeler/topics.py`; the prompt is in `src/policywinds_labeler/label.py`.

## What the code does

| Module | Job |
|---|---|
| `topics` | Topics, subtags, definitions, keywords, scope exclusions and the measure of each topic |
| `prefilter` | Keyword filter on the title (which bills a model reads) and the text windows it is shown |
| `text` | Plain text from official HTML, XML or PDF; struck-out law is kept and marked `[deleted: ...]` |
| `label` | The prompt, the request message and the checks on every answer: quotes must appear in the text, custody labels need the topic's keywords in the title, duplicate tags are merged, stray foreign-script text is retried |
| `llm` | Answer schema, provider adapters (Anthropic, OpenAI, xAI), prices |

The model never decides alone what counts: an answer whose quote is not in the bill is marked unverified, and for
topics that need it (`REQUIRES_KEYWORD`: custody and support), a label is dropped when the topic's keywords are not
in the bill's title or description.

## Use it

```sh
uv sync
uv run pytest                         # unit tests, no keys needed
cd eval && uv run python fetch_texts.py && ANTHROPIC_API_KEY=... uv run python run_eval.py
```

```python
from policywinds_labeler.label import system_prompt, user_message, verify_quotes
from policywinds_labeler.llm.providers import get_provider
```

Keys are read from `ANTHROPIC_API_KEY`, `OPENAI_API_KEY` or `XAI_API_KEY`, or from any function passed as
`get_provider(name, secret=...)`.

## The eval

`eval/` holds checks with known answers on real bills, the official texts, and the harness. See
`eval/README.md`. Any change to the prompt, keywords or model must run it first.

## How the model was chosen

PolicyWinds labels with Claude Sonnet 5.5 (`claude-sonnet-5-5`), chosen from a test of 13 models on the eval set
(2026-11-21): it had 2 real errors on 44 checks, at 3.3 seconds a bill. Any model the provider adapters support
can be compared with `eval/run_eval.py --provider P --model M`.

## Versions

The package version follows semantic versioning, read for labels:

- **Minor** (0.1.0 -> 0.2.0, while the version is 0.x): anything that can change labels. That is the prompt, topics,
  definitions, keywords, scope exclusions, the answer schema, or the checks on answers.
- **Patch** (0.1.0 -> 0.1.1): changes that cannot change labels, such as provider adapters, prices, docs or tests.

The prompt has its own number, `policywinds_labeler.label.PROMPT_VERSION`. Every release note names it, and
`eval/results/` keeps that release's eval output.

| Release | Prompt | Eval (claude-sonnet-5-5) |
|---|---|---|
| 0.1.0 | v9 | `eval/results/0.1.0.txt` |
| 0.2.0 | v10 | `eval/results/0.2.0.txt` |
| 0.3.0 | v10 | same as 0.2.0: only the quote check changed (short citation quotes and quotes joined with "..." can now pass), not the labels |

## Wrong label?

Open an issue with the bill, the tag, the label and what you think it should be, quoting the bill's text.
Confirmed cases become new eval checks.

## License

Code: Apache-2.0 (`LICENSE`). Eval checks: CC BY 4.0 (`eval/LICENSE`).
Copyright 2026 Kernelcraft LLC, which publishes PolicyWinds.
