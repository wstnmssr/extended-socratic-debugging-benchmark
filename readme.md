# Extended Socratic Debugging Benchmark

Here is [the original project](https://github.com/taisazero/socratic-debugging-benchmark) that this work extends from. You can also view the [readme from that repository](previous_readme.md).

## What is different?

This repository contains work that extends from the project linked above in three main ways:

- Add a class for [Ollama predictions](inference/llama_inference.py)
- Create a Java counterpart to the conversation dataset in Python
  - Both datasets are in the `scoratic_debugging_benchmark` directory
- Train a reward model forked from OPT1.3b
  - Scripts and data used for this are in the `reward_model` directory
  - Due to file size restrictions, some files are compressed as .zip files. If you intend to use the reward model, ensure all relevant directories are uncompressed.

The `requirements.txt` file has been updated to include packages that are necessary to run the reward model. As such, the instructions are the same as the previous readme: Create a new virtual Python environment called `socratic_env` using Anaconda or virtualenv. Then, activate the environment and install the required packages using the following command:

``` bash
pip install -r requirements.txt
```

For any questions specific to these extensions to the previous work, please contact [westin.musser@colostate.edu](mailto:westin.musser@colostate.edu)
