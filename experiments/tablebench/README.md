# How to Reproduce MATA Experiment Results on the `Tablebench` Dataset

## TableBench

[TableBench](https://tablebench.github.io/) is a comprehensive benchmark for complex table QA, spanning 18 subcategories across fact checking, numerical reasoning, data analysis, and visualization. Tables are sourced from diverse domains such as finance, sports, and science. The benchmark emphasizes real-world complexity and supports multiple reasoning paradigms including TCoT, SCoT, and PoT.  In our experiments, as our focus is on entity-type answers, we excluded table-to-text samples from TableBench, which provide sentence-level ground truth. Therefore, 693 out of 886 TableBench test samples are included in our evaluation.


By following the procedure below, you can reproduce the experimental results on the Tablebench dataset, which is one of the benchmark datasets used in our paper.

---


## MATA scheduler Checkpoint
You can download the MATA scheduler checkpoint from the following [link](https://drive.google.com/file/d/1Yxz5xZMOBeQyPc0lK0ZDCQv1VDtUVh-z/view?usp=drive_link).

## MATA confidence checker Checkpoint
You can download the MATA confidence checker checkpoint from the following [link](https://huggingface.co/snu-aidas/MATA_confidence_checker).

---
## How to Use

**1. Clone this repository using the web URL.**
```bash
git clone https://github.com/AIDASLab/MATA.git
```
**2. To use MATA, you need to install [Ollama](https://ollama.com/). Please run the following code in your local environment. Our code is designed to be used on Linux systems.**
```bash
curl -fsSL https://ollama.com/install.sh | sh
```
**3. Place [the scheduler checkpoint](https://drive.google.com/file/d/1Yxz5xZMOBeQyPc0lK0ZDCQv1VDtUVh-z/view?usp=drive_link) inside the [`scheduler` folder](https://github.com/AIDASLab/MATA/tree/main/scheduler).**

**4. Run the following code.**
```bash
ollama serve
```
**5. Check whether the model you want to use is supported by Ollama on the [official Ollama website](https://ollama.com/search), then pull the corresponding model using the code below. (The model name `qwen2.5:32b-instruct` in the code is just an example.)**
```bash
ollama pull qwen2.5:32b-instruct
```
The format matcher in `utils/FM_inference.py` uses `qwen2.5:0.5b-instruct-q8_0`, so please also pull it:

```bash
ollama pull qwen2.5:0.5b-instruct-q8_0
```

**6. Move the `MATA_tablebench.py` file and the `Tablebench_loader.py` file to the [main](https://github.com/AIDASLab/MATA/tree/main) folder.**

**7. If you want to change the Ollama model, update the following locations consistently:**

- `MATA_tablebench.py`: every `ChatOllama(model=...)` value.
- `utils/adjust_context.py`: the fallback model used inside `llm_adjusted_context`.
- `utils/adjust_context.py`: the Hugging Face tokenizer name in `measure_and_adjust_context(model_name=...)`.
- `utils/FM_inference.py`: the small format-matcher model used when the final answer is longer than 100 characters.



**8. Our code was developed in an [Anaconda](https://www.anaconda.com/) environment. Please run the code below to create a new virtual environment. This will make it easy to install the libraries required for MATA.**
```bash
conda env create -f ./langchain.yml
```


**9. Run the following code.**
```bash
python MATA_tablebench.py --config config.yaml
```

**10. If you do not want to use the scheduler or want to increase the number of self-refinement iterations, you can either modify the `config.yaml` file or run the code as shown below.**
```bash
python MATA_tablebench.py --config config.yaml --Use_Scheduler False --N 5
```

**Notes:** This repository provides code for using MATA with the `qwen2.5:32b-instruct` model. If you want to use a different model, please follow the guidelines mentioned above.

