# Prompts Repository

Welcome to the Prompts repository! This project is dedicated to researching and testing **prompt injection techniques** in Large Language Models (LLMs) using the **Giskard framework**.

## Repository Overview

This repository contains several Python files, each focused on testing different LLMs for prompt injection vulnerabilities. The tests leverage **Giskard's scan() method** and two separate **Giskard detectors** to identify prompt injection risks.

### Files & Implementations

- **Mist1.py**: Runs **Mistral 7B locally** and applies Giskard detectors - LLMPromptInjectionDetector() and LLMCharsInjectionDetector().
- **Ollama2_2.py**: Runs **Ollama 7B locally** and applies Giskard detectors - LLMPromptInjectionDetector() and LLMCharsInjectionDetector().
- **Llama70B.py**: Runs **LLaMa 70B via the Together API** and applies the **scan()** method.
  - This code can be easily modified to test any model supported by the **Together API**.
- **Mist2.py & Ollama2_1.py**: Additional prompt injection test cases.

## Customization

- The `Llama70B.py` file can be adapted for **any model** supported by **Together API** by modifying the API request parameters.
- The repository structure allows easy integration of additional models for **prompt injection testing**.

## Acknowledgements

This project leverages the **Giskard framework** for AI model safety testing and evaluation. We appreciate the contributions from the AI security community in developing robust prompt injection detection methods.

---

*This repository is continuously evolving. Check back for updates and new prompt injection techniques!*

