Malayalam evaluation fixture pack
================================

This directory is the input pack for `backend/videos/eval_malayalam_wer.py`.

What belongs here
-----------------

- Real Malayalam audio clips with human-verified reference transcripts
- Reference transcripts in Malayalam Unicode
- Coverage for the known production failure modes:
  - soft-spoken sections
  - continuous speech with no silence
  - authentic Malayalam + English code-switching
  - volume spikes / field-recording variation
  - background noise or music-under-speech

Current state
-------------

The repo currently contains the fixture manifest and case definitions, but not
the actual benchmark audio files or their verified reference transcripts yet.
That is intentional: the harness is ready, but it should not pretend to have
ground-truth fixtures that are not present in source control.

To populate this pack
---------------------

For each case listed in `manifest.json`, add:

- the audio file at the relative `audio_file` path
- the verified reference transcript at the relative `reference_file` path

Reference transcript rules
--------------------------

- Malayalam Unicode only for Malayalam speech
- Keep real spoken English terms only when actually spoken
- Human-verified, not machine-generated
- Preserve meaningful spoken structure, not summary text
