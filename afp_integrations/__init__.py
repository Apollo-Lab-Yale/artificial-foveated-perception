"""AFP (Artificial Foveated Perception) integration with VLA policies.

    afp_utils.py            AFP model loading and mask-pooling helpers.
    attention_aux_loss.py   Auxiliary attention loss that aligns a policy's
                            attention over image tokens with the AFP mask, the
                            attention-capture hook, and the projected-gradient
                            (PCGrad) combiner for action / auxiliary gradients.
"""
