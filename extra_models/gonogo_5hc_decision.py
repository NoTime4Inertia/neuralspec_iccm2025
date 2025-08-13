import nengo
import numpy as np
import nengo_spa as spa
from time import sleep

d0 = 128
d1 = 128
d2 = 128
d3 = 128
d4 = 128
d5 = 128

obs_vocab = spa.Vocabulary(d0)
hc_vocab = spa.Vocabulary(d1)

stm_vocab1 = spa.Vocabulary(d2)
stm_vocab2 = spa.Vocabulary(d2)

act_vocab = spa.Vocabulary(d4)
fpn_vocab1 = spa.Vocabulary(d3)
fpn_vocab2 = spa.Vocabulary(d3)
fpn_unseg_vocab = spa.Vocabulary(d3)
assoc_ppc_vocab = spa.Vocabulary(d5)
ppc_vocab = spa.Vocabulary(d5)

obs_vocab.populate(
    "VOW; CONS"
    )

hc_vocab.populate(
    "VOW; CONS; "
    "GO; NOGO"
    )
    
assoc_ppc_vocab.populate(
    "GO; NOGO; "
    "VOW=GO; CONS=NOGO"
    )

ppc_vocab.populate(
    "GO; NOGO"
    )

fpn_unseg_vocab.populate(
    "GO; "
    "NOGO"
    )
    
def cue_input(t):
    sequence = ["VOW", "CONS", "VOW", "CONS", "CONS", "VOW", "CONS", "VOW"]
    if sequence == "C":
        idx = int((t // (5.0 / len(sequence))) % len(sequence))
    else:
        idx = int((t // (10.0 / len(sequence))) % len(sequence))
    return sequence[idx]

model = spa.Network()
with model:
    
    stim = spa.Transcode(cue_input, output_vocab=obs_vocab)
    obs = spa.State(obs_vocab, neurons_per_dimension = 10)
    
    hc_obs = spa.State(hc_vocab, neurons_per_dimension = 10)
    hc_ppc = spa.State(hc_vocab, neurons_per_dimension = 10)
    hc = spa.State(hc_vocab, neurons_per_dimension = 5)
    hc_clean = spa.ThresholdingAssocMem(threshold=0.0, input_vocab=hc_vocab, mapping = hc_vocab.keys())    

    assoc_ppc = spa.State(assoc_ppc_vocab, neurons_per_dimension = 10)
    ppc = spa.State(ppc_vocab, neurons_per_dimension = 10)

    
    # Feed forward
    stim >> obs
    hc >> hc_clean
    spa.translate(assoc_ppc, ppc_vocab) >> ppc
    #spa.translate(assoc_fpn, fpn_unseg_vocab) >> fpn_unseg

    # Backprop
    spa.translate(obs, assoc_ppc_vocab) >> assoc_ppc
    spa.translate(obs, hc_vocab) >> hc_obs
    spa.translate(ppc, hc_vocab) >> hc_ppc
    spa.translate(obs, hc_vocab)*~hc_obs*hc_ppc >> hc