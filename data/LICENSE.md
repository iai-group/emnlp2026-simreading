# Data License and Attribution

The data in this repository is **not** released under a single license.
Code is licensed separately — see the top-level [`LICENSE`](../LICENSE)
(CC BY 4.0, © Krisztian Balog).

## Questions, categories, and aggregated student responses

`data/questions.json` (the questions, answer options, and gold answers —
excluding the reading texts themselves), `data/question_categories.csv`,
and everything under `data/aggregates/human/` (the aggregated,
anonymized student-response statistics) are:

> © Nasjonalt lesesenter (The Norwegian Reading Centre), University of
> Stavanger. Licensed under
> [CC BY-NC-ND 4.0](https://creativecommons.org/licenses/by-nc-nd/4.0/).

## Reading texts

The reading texts embedded in `data/questions.json` (the `text` field of
each item) are third-party copyrighted material, each individually
licensed by its own rights holder. Per-text attribution — rights holder,
license, and a link to the license deed — is listed in
[`text_licenses.md`](text_licenses.md), keyed by `sanity_text_id`
(matching `data/questions.json`).

Two licenses are used across the texts:

- [CC BY-NC-ND 4.0](https://creativecommons.org/licenses/by-nc-nd/4.0/) — Attribution-NonCommercial-NoDerivatives
- [CC BY-SA 3.0](https://creativecommons.org/licenses/by-sa/3.0/) — Attribution-ShareAlike

**Coverage.** `text_licenses.md` currently covers 154 of the 156 texts in
`data/questions.json`. Attribution for the remaining 2 texts
(`HT_14` / "Vulkaner" and `HT_161` / "Teknisk museum") is not yet
available and will be added once confirmed; until then, treat those two
texts as unlicensed for redistribution purposes.

## Precomputed simulation aggregates

`data/aggregates/simulation/` contains outputs of the LLM-based
simulators described in the paper (not third-party material); these are
released under the same CC BY 4.0 terms as the code — see the top-level
[`LICENSE`](../LICENSE).
