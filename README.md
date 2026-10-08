# Canopy

Canopy is a non-autoregressive Conformer-CTC speech recognition library. One install loads every published variant from the Hugging Face Hub.

```bash
pip install torch torchaudio soundfile
pip install git+https://github.com/Proxima-AI-Co/canopy.git
hf auth login
canopy-transcribe clip.wav --model canopy-m --language urd
```

```python
from canopy import Canopy

asr = Canopy.from_pretrained("canopy-m")
print(asr.transcribe("clip.wav", language="urd"))
print(asr.transcribe("clip.wav", language="snd"))
```

`canopy-m` is the Urdu and Sindhi model. A later variant is the same call with another name or a full Hub id:

```python
asr = Canopy.from_pretrained("ProximaAI/Canopy-M")
asr = Canopy.from_pretrained("ProximaAI/Canopy-S")   # when that repository exists
```

If a repository contains more than one checkpoint, pass `filename=`. A path to a local `.pt` file loads that file and does not contact the Hub.

## Variants

| Name | Hub repository | Languages | Checkpoint |
|---|---|---|---|
| `canopy-m` | [ProximaAI/Canopy-M](https://huggingface.co/ProximaAI/Canopy-M) | Urdu `urd`, Sindhi `snd` | `canopy-m.pt` |

`Canopy-M` is private. `hf auth login` needs a token that can read it. Language names `ur` / `urdu` and `sd` / `sindhi` are accepted for that model. Decoding is greedy CTC. Audio is resampled to 16 kHz mono. Files longer than 30 seconds are transcribed in chunks.

To publish another variant, upload an inference checkpoint to a Hub repo and add one line to `VARIANTS` in [`canopy/inference/hub.py`](canopy/inference/hub.py):

```python
VARIANTS = {
    "canopy-m": "ProximaAI/Canopy-M",
}
```

Until that line exists, `Canopy.from_pretrained("Org/Repo")` still loads the repo. The default weight file is `<repo-name>.pt` at the repository root.

## Canopy-M

57.9M parameters, 12 Macaron blocks, width 384, 6 rotary attention heads. Development-set greedy CTC, no language model:

| | Urdu | Sindhi | Macro |
|---|---:|---:|---:|
| CER | 0.132 | 0.107 | 0.119 |
| WER | 0.267 | 0.271 | 0.269 |

The held-out test split has not been scored. The model card on the Hub has the full comparison.

## Command line

```bash
canopy-transcribe clip.wav --model canopy-m --language urd
canopy-transcribe clip.wav --model ProximaAI/Canopy-M --language snd --device cpu
canopy-transcribe a.wav b.wav --model canopy-m --language urdu --decoder beam
```

## Licence

Code in this repository is Apache-2.0. See [LICENSE](LICENSE) and [NOTICE](NOTICE). Weights are licensed with the Hub repository that ships them.
