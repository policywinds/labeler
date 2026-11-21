# Eval set

A fixed set of checks with known answers. A prompt, keyword or model change must run it before it ships.

| File | What it is |
|---|---|
| `labels.jsonl` | One check per line: a bill (`state`, `number`, `year`), a tag, the expected `direction`, who checked it and why. |
| `bills.jsonl` | One record per bill: title, description, sponsors, the text version used and where its text comes from. |
| `texts/` | Official bill texts, as plain text. |
| `fetch_texts.py` | Downloads the texts that are not stored here, from the legislature's own site. |
| `run_eval.py` | Labels each bill once and scores every check. |

Directions: `expands` or `restricts` (the topic's fixed measure, see `DIRECTION_AXES` in
`policywinds_labeler/topics.py`), `neutral` (no change either way), or `absent` (the bill must not get any tag of
that topic: a scope check).

## Run it

```sh
python fetch_texts.py                      # once; texts marked "manual" are listed with their URL
ANTHROPIC_API_KEY=... python run_eval.py   # or --provider openai|xai --model M with that provider's key
```

A check passes when the exact tag has the expected direction, or when the model filed the bill under another
subtag of the same topic and that topic's non-neutral directions all agree. **Real errors** count only a wrong
direction, or a direction where none belongs (and the reverse). "No change" and "not this topic" are the same
answer, because both mean the bill does not move the measure. A model is acceptable at 0-3 real errors.

## Where the texts come from

- `committed`: stored in `texts/`. Texas Legislature Online, the California Legislature's bulk export, the New
  York Assembly and GovInfo.
- `fetch`: downloaded by `fetch_texts.py` from the official URL (Florida Senate), 3 seconds apart.
- `manual`: the state publishes no document that can be fetched directly (Virginia). Save the text by hand as the
  file named in `bills.jsonl`; until then the check is skipped.

## Reporting a wrong label

Open an issue with the bill, the tag, what the label says and what you think it should say, quoting the bill's
text. Confirmed cases are added here as new checks.

License: see `LICENSE` in this folder.
