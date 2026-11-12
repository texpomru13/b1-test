Rejected-quote capture (diagnosis only, nothing tuned on these cases): missed answerable
cases of the thinking-off run were re-asked inside the API container with
`eval/incontainer/capture_quotes.py`, which wraps `grounding.locate_quote` of the
pre-v2 matcher and classifies each rejected quote. Classes `whitespace_only` motivated
quote matching v2 (`app/qa/grounding.py`, `QUOTE_MATCH_VERSION`).
