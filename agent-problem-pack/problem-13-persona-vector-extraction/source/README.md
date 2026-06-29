# Persona Vectors (condensed methodology)

Persona Vectors are a method for **monitoring** and **controlling** character traits
(e.g. "evil", "sycophantic", "hallucinating") in language models. A persona vector is a
single direction in a model's residual stream (activation space) that corresponds to a
trait. Once you have that direction you can:

- **Monitor**: project a model's activations onto the vector to measure how strongly the
  trait is currently expressed.
- **Control**: add or subtract the vector from activations to amplify or suppress the
  trait — at inference time, or preventatively during fine-tuning.

The reference implementation lives in `generate_vec.py` (extraction) and
`activation_steer.py` (control). This document summarizes how they fit together.

## Pipeline overview

1. **Trait artifacts.** Each trait ships positive and negative *system prompts*, a set of
   evaluation questions, and judge prompts. Positive prompts instruct the model to exhibit
   the trait ("You are an evil assistant..."); negative prompts instruct the opposite.

2. **Elicit + judge.** `eval.eval_persona` runs the model under the positive and negative
   system prompts to produce `(prompt, answer)` pairs, and an LLM judge scores each answer
   for the trait and for `coherence`. Results are written to CSV files
   (`*_pos_instruct.csv`, `*_neg_instruct.csv`) with `prompt`, `answer`, `<trait>`, and
   `coherence` columns.

3. **Filter effective examples.** `get_persona_effective(...)` keeps only the contrastive
   pairs where the positive prompt actually elicited the trait (`trait >= threshold`) and
   the negative prompt suppressed it (`trait < 100 - threshold`), with both responses
   staying coherent (`coherence >= 50`). This removes noisy pairs before averaging.

4. **Extract activations.** `get_hidden_p_and_r(...)` runs each `prompt + answer` text
   through the model with `output_hidden_states=True` and, per layer, records three
   summaries: the mean over prompt tokens (`prompt_avg`), the mean over response tokens
   (`response_avg`), and the last prompt token (`prompt_last`).

5. **Mean difference -> vector.** For each layer the persona vector is the mean positive
   activation minus the mean negative activation:

   ```
   vector[layer] = mean(activation | positive) - mean(activation | negative)
   ```

   This is computed for each of the three summaries, producing
   `<trait>_prompt_avg_diff.pt`, `<trait>_response_avg_diff.pt`, and
   `<trait>_prompt_last_diff.pt`. Each saved tensor has shape `[num_layers x hidden_dim]`.
   **The paper primarily uses `response_avg_diff`** (the difference of response-token
   activations).

## Monitoring (projection)

To detect how strongly a trait is present, take a layer's persona vector and compute the
dot product (projection) of new activations onto it (`eval.cal_projection`). A larger
projection means the trait direction is more active.

## Control (steering)

`activation_steer.py` provides `ActivationSteerer`, a context manager that registers a
**forward hook** on one transformer block and adds `coeff * steering_vector` to that
block's hidden-state output:

- `coeff` (a.k.a. `coef`): steering strength. Positive amplifies the trait, negative
  suppresses it.
- `layer_idx` / `layer`: which transformer block to hook.
- `positions`: which token positions to modify — `"all"`, `"prompt"`, or `"response"`.

`ActivationSteererMultiple` applies several such edits at once.

### Inference-time steering

Wrap a generation call in the `ActivationSteerer` context manager with a chosen
`vector_path`, `layer`, and `coef`. The hook perturbs activations only while generating,
without changing the weights.

### Training-time steering (preventative)

During fine-tuning you can steer along the persona vector so the model does not need to
*learn* the trait to fit the data — this "preventative steering" reduces unwanted trait
drift from the training set. The training config selects:

- `type`: `"steer"` (preventative steering, scaled by `steering_coef`) or `"ablate"`
  (the CAFT / directional-ablation variant that projects the direction out).
- `steering_coef`: strength for the `"steer"` type.
- `layers`: which transformer layers to apply it to.

## Key files

- `generate_vec.py` — extract and save persona vectors (steps 3-5 above).
- `activation_steer.py` — `ActivationSteerer` / `ActivationSteererMultiple` for control.
- `eval/eval_persona.py` — elicit responses and judge them (step 2).
- `eval/cal_projection.py` — projection-based monitoring.
- `training.py` — fine-tuning with optional training-time steering.
