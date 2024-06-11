# LLM Fine-tuning using PEFT

---

## 1. Introduction :open_book:

With this project we will see how you can fine-tune some parameters of your Large Language Model without having to retrain the whole model (which would be very long and very costly). 

If you wish to reuse this code and adapt it to your needs, I highly encourage you to use a Google Colab notebook with below parameters. Otherwise, you may have trouble loading and saving the model (250M parameters is quite heavy for a *standard* laptop).

<img width="549" alt="Screenshot 2024-06-10 at 18 19 32" src="https://github.com/julienguyet/peft_fine_tuning/assets/55974674/4d170554-a8ef-41b2-8953-cb0399f07487">

---

## 2. Loading the model :telescope:

We will work with the famous "dialogsum" dataset. It is made of dialogues that could happend between humans. All conversations have corresponding summaries.
You can obtain more information [here](https://huggingface.co/datasets/knkarthick/dialogsum). Below is the sample code to import it:

```
huggingface_dataset_name = "knkarthick/dialogsum"
dataset = load_dataset(huggingface_dataset_name)
```

Regarding the model, we will use Flan-T5: it was designed by Google's team and first released in the paper *Scaling Instruction-Finetuned Language Models* (by HW Chung et al., 2022). Paper is available [here](https://arxiv.org/pdf/2210.11416) and you can consult module's documentation at HuggingFace dedicated [page](https://huggingface.co/docs/transformers/en/model_doc/flan-t5).

```python
model_name='google/flan-t5-base'

original_model = AutoModelForSeq2SeqLM.from_pretrained(model_name, torch_dtype=torch.bfloat16)
tokenizer = AutoTokenizer.from_pretrained(model_name)
```

Note that we also used the associated Tokenizer to avoid any mismatch between the input/output format expectations of the model.
If we check the size of the model we get the below results:

```
trainable model parameters: 247577856
all model parameters: 247577856
percentage of trainable model parameters: 100.00%
```
