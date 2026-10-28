# LLMs From Scratch

This repository is a hands-on, end-to-end study of how large language models
(LLMs) work. It builds the main pieces of a GPT-style language model in
PyTorch, starting with text and token IDs and progressing through attention,
transformer blocks, training, text generation, and loading pretrained
parameters.

The central model is the **124-million-parameter GPT-2 model**. The project
first implements its components from scratch to make the architecture
understandable, then loads the weights released by OpenAI into the PyTorch
implementation. The notebooks are intended to be read and run in order; they
are instructional experiments, not a packaged library or a production
training system.

## What you will learn

- How text is split into tokens, mapped to token IDs, and arranged into
  input/target sequences.
- How token embeddings and positional embeddings represent a sequence.
- How scaled dot-product self-attention works, including causal masks,
  dropout, and multiple attention heads.
- How GPT transformer blocks combine layer normalization, feed-forward
  networks, GELU, residual connections, and attention.
- How to assemble a GPT-2-style model in PyTorch and inspect its output.
- How to calculate cross-entropy loss and perplexity, train with an
  optimizer, evaluate on validation data, and save/load PyTorch checkpoints.
- How temperature and top-k sampling affect generated text.
- How to download OpenAI's original GPT-2 124M checkpoint and transfer its
  parameters into the PyTorch model.

## Repository layout

```text
.
|-- README.md
|-- the-verdict.txt
|-- Practise on Small Dataset/
|   `-- Vector_Embedding.ipynb
|-- Stage 1/
|   |-- 1.1_Data_Preprocessing_And_Sampling.ipynb
|   |-- 1.2_Attention_Mechanism.ipynb
|   `-- 1.3_LLM_Architecture.ipynb
|-- Stage_2/
|   |-- Notebooks/
|   |   |-- 2.1_Training_Loop.ipynb
|   |   `-- 2.2_Load_Pretrained_Weights.ipynb
|   |-- Scripts/
|   |   `-- gpt_download3.py
|   `-- Checkpoints/
|       |-- model.pth
|       `-- model_and_optimizer.pth
`-- gpt2/
    `-- 124M/                 # Local OpenAI checkpoint and tokenizer files
```

## Learning path

### 0. Optional: word-vector practice

[`Practise on Small Dataset/Vector_Embedding.ipynb`](<Practise on Small Dataset/Vector_Embedding.ipynb>)
uses the pretrained Google News Word2Vec vectors through Gensim. It is a
separate introductory exercise, not the GPT-2 model or its weights.

### 1. Prepare text and understand tokens

[`Stage 1/1.1_Data_Preprocessing_And_Sampling.ipynb`](<Stage 1/1.1_Data_Preprocessing_And_Sampling.ipynb>)
uses [`the-verdict.txt`](the-verdict.txt) to explore simple tokenizers, GPT-2
byte-pair encoding, special tokens, token IDs, input/target pairs, a PyTorch
dataset/data loader, and embeddings.

### 2. Build attention

[`Stage 1/1.2_Attention_Mechanism.ipynb`](<Stage 1/1.2_Attention_Mechanism.ipynb>)
develops self-attention step by step, then adds trainable query/key/value
projections, causal masking, dropout, and multi-head attention.

### 3. Assemble the GPT architecture

[`Stage 1/1.3_LLM_Architecture.ipynb`](<Stage 1/1.3_LLM_Architecture.ipynb>)
implements layer normalization, GELU and feed-forward layers, shortcut
(residual) connections, transformer blocks, and the complete GPT model in
PyTorch.

### 4. Train and generate text

[`Stage_2/Notebooks/2.1_Training_Loop.ipynb`](<Stage_2/Notebooks/2.1_Training_Loop.ipynb>)
builds the training and validation loops, evaluates loss, generates samples,
and demonstrates saving and restoring model and optimizer state. The included
GPT-2 configuration has 12 layers, 12 attention heads, and 768-dimensional
embeddings. Its context length is set to 256 for the notebook exercise
(GPT-2's original context length is 1,024). The training example learns from
the included short text; it does not reproduce the original GPT-2 training.

### 5. Load OpenAI GPT-2 weights

[`Stage_2/Notebooks/2.2_Load_Pretrained_Weights.ipynb`](<Stage_2/Notebooks/2.2_Load_Pretrained_Weights.ipynb>)
downloads the original OpenAI TensorFlow checkpoint, reads its parameters,
maps them onto the matching PyTorch model layers, and uses the GPT-2 tokenizer
to generate text. This is a separate workflow from training the model from
random initialization.

## Setup

Use Python in an environment compatible with your installed PyTorch and
TensorFlow versions. No dependency lockfile is included. From the repository
root, create and activate a virtual environment, then install the notebook
dependencies:

```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install jupyter ipykernel torch tiktoken numpy matplotlib requests tensorflow tqdm
```

For GPU-enabled PyTorch, choose the installation command that matches your
CUDA setup from the official [PyTorch installation guide](https://pytorch.org/get-started/locally/).
The vector-embedding practice notebook additionally needs Gensim:

```powershell
python -m pip install gensim
```

Open the repository in VS Code, select this virtual environment as the
notebook kernel, and run cells from top to bottom. The notebooks use relative
paths, so set the notebook working directory to the folder containing the
notebook:

- For the Stage 1 notebooks, use `Stage 1/`.
- For `2.1_Training_Loop.ipynb` and `2.2_Load_Pretrained_Weights.ipynb`, use
  `Stage_2/Notebooks/`.
- For the optional vector-embedding notebook, use
  `Practise on Small Dataset/`.

## Download and use OpenAI's GPT-2 124M weights

The pretrained-weight notebook uses the original GPT-2 TensorFlow checkpoint
files published by OpenAI. The downloader stores them under
`gpt2/124M/`, including the model checkpoint, `hparams.json`, and GPT-2
tokenizer files (`encoder.json` and `vocab.bpe`). The checkpoint data is large;
allow several hundred megabytes of disk space and a working internet
connection. The downloader also needs TensorFlow to read and convert the
checkpoint.

1. Install the dependencies above and select the environment as the notebook
   kernel.
2. Open `Stage_2/Notebooks/2.2_Load_Pretrained_Weights.ipynb` and run it from
   its containing folder. Its import cell loads the helper in
   `Stage_2/Scripts/gpt_download3.py`.
3. Use the following call (the downloader accepts the **parent** model
   directory and adds the `124M` folder itself):

   ```python
   settings, params = download_and_load_gpt2(
       model_size="124M",
       models_dir="../../gpt2",
   )
   ```

   The supported OpenAI checkpoint sizes in the helper are `124M`, `355M`,
   `774M`, and `1558M`; this project is configured and documented around
   `124M`.
4. Create the same-size PyTorch GPT model, transfer the loaded parameters,
   and switch to evaluation mode:

   ```python
   gpt = GPTModel(GPT_CONFIG_124M)
   load_weights_into_gpt(gpt, params)
   gpt.eval()
   ```

   The model configuration and the `load_weights_into_gpt` mapping are
   defined in the notebook. Use its GPT-2 tokenizer setup
   (`tiktoken.get_encoding("gpt2")`) to encode prompts and decode generated
   token IDs.
5. Continue with the notebook's generation example. The weight transfer
   initializes the PyTorch model with pretrained parameters; it does not
   train or fine-tune those parameters.

The helper uses HTTPS with certificate verification enabled. Downloaded
checkpoints and locally saved PyTorch files are data artifacts rather than
source code; keep them out of commits unless you intentionally want to
version and distribute those files.

## Checkpoints and outputs

The training notebook demonstrates saving a model `state_dict` as
`model.pth`, and saving both model and optimizer state in
`model_and_optimizer.pth`. It also demonstrates how to restore those states.
The repository includes example checkpoint files under `Stage_2/Checkpoints/`;
new runs may save output in the notebook's current working directory unless
you change the save path.

These are PyTorch training checkpoints. They are distinct from OpenAI's
original TensorFlow checkpoint files under `gpt2/124M/`. The latter must be
loaded and mapped into the PyTorch architecture before that model can use
them.

## Notes

- Run notebook cells in sequence: later cells reuse classes and variables
  defined earlier in the same notebook.
- If a file is not found, verify the notebook kernel's working directory
  before changing the relative path.
- The 124M configuration is substantial for CPU-only training. Use a smaller
  batch size or fewer training steps if memory or runtime is limited.
- Generated text is stochastic when sampling is enabled. Results depend on
  the model state, prompt, and decoding settings.
- Check the terms associated with the model and source text before
  redistributing downloaded weights or generated datasets.