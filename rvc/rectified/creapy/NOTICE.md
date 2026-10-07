# creapy

`model_ALL.csv` is the training data of the `all` gender model from
[creapy](https://gitlab.tugraz.at/speech/creapy), a Python tool for the automatic
detection of creak in conversational speech from the Speech Communication Laboratory
at Graz University of Technology. It is copied unchanged.

`rvc/rectified/creak.py` reimplements creapy's classification path with its default
settings: 40 ms Hann blocks every 10 ms, zero-crossing-rate and short-term-energy
exclusion of unvoiced blocks, Praat HNR, jitter, shimmer, H1-H2 and mean F0 per block,
and a 99-tree random forest (seed 42) fitted on this file with median imputation.

The copy this was taken from (bundled with CreakVC) does not state a licence; its
`setup.cfg` lists BSD 3-Clause only in commented-out lines. Check the upstream
repository before redistributing it.
