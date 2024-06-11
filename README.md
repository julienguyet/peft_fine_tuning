# LLM Fine-tuning using PEFT

---

## 1. Introduction :open_book:

With this project we will see how you can fine-tune some parameters of your Large Language Model without having to retrain the whole model (which would be very long and very costly). 

If you wish to reuse this code and adapt it to your needs, I highly encourage you to use a Google Colab notebook with below parameters. Otherwise, you may have trouble loading and saving the model (250M parameters is quite heavy for a *standard* laptop).

<img width="549" alt="Screenshot 2024-06-10 at 18 19 32" src="https://github.com/julienguyet/peft_fine_tuning/assets/55974674/4d170554-a8ef-41b2-8953-cb0399f07487">

---

## 2. Loading the Model :telescope:

We will work with the famous "dialogsum" dataset. It is made of dialogues that could happend between humans. All conversations have corresponding summaries.
You can obtain more information [here](https://huggingface.co/datasets/knkarthick/dialogsum). Below is the sample code to import it:

```python
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
all model parameters: 247577856
percentage of trainable model parameters: 100.00%
```

---

## 3. Base Model Performance :weight_lifting:

Our goal is to generate summary of dialogues. Before trying to update any parameters of the model, we should check its current performance. 

First, we will design a prompt:

```python
index = np.random.randint(100)
dialogue = dataset["train"][index]["dialogue"]
summary = dataset["train"][index]["summary"]

prompt = f"""

Summarize the following conversation:
{dialogue}

Summary:
"""
```

Which results into the below prompt:
```
Input Prompt:


Summarize the following conversation:
#Person1#: What kind of job do you intend to do?
#Person2#: I want to do some management job since I have three-year's work history.
#Person1#: What are your plans if you were hired?
#Person2#: I would apply my specialty and experience to my job and gradually move up to the management level in this company.

Summary:
```

Then, we can call the model using the HuggingFace library:

```python3
input = tokenizer(prompt, return_tensors="pt")
output = tokenizer.decode(original_model.generate(input["input_ids"],
                                                max_new_tokens=200)[0],
                                                skip_special_tokens=True)
```

This code will feed the model with a prompt (the input variable) and we will decode model's predictions using the tokenizer.decode() function. Which gives us:

```
Input Prompt:


Summarize the following conversation:
#Person1#: What kind of job do you intend to do?
#Person2#: I want to do some management job since I have three-year's work history.
#Person1#: What are your plans if you were hired?
#Person2#: I would apply my specialty and experience to my job and gradually move up to the management level in this company.

Summary:

----------------------------------------------------------------------------------------------------
Baseline Human Summary:
#Person2# tells #Person1# #Person2#'s ideal job and the job plan if hired.
----------------------------------------------------------------------------------------------------
LLM Summary - Zero Shot:
Ask the person to describe their career goals.
```

As we can see, the model summary is quite different from the human baseline. Let's compute the ROUGE score to get an idea of how well the model is performing.

```
ORIGINAL MODEL:
{'rouge1': 0.0970873786407767, 'rouge2': 0.0, 'rougeL': 0.0970873786407767, 'rougeLsum': 0.0970873786407767}
```

ROUGE scores are computed like below:

$$ \text{ROUGE-1} = \frac{\text{Unigram Matches}}{\text{Total number of unigram in reference}} $$

$$ \text{ROUGE-2} = \frac{\text{Unigram Matches}}{\text{Unigrams in Output}} $$

$$ \text{ROUGE-L} = \frac{\text{Length of longest common subsequence}}{\text{Total number of words in reference summaries}} $$

$$ \text{ROUGE-Lsum} = \frac{\text{Total length of overlapping summary}}{\text{Total length of reference summaries}} $$

As per [Wikipedia](https://en.wikipedia.org/wiki/N-gram), an "n-gram is a sequence of n adjacent symbols in particular order. The symbols may be n adjacent letters (including punctuation marks and blanks), syllables, or rarely whole words found in a language dataset; or adjacent phonemes extracted from a speech-recording dataset, or adjacent base pairs extracted from a genome".

In other words, the higher the score the better. An optimal score would be a ROUGE score equal to 1. As we can see here, we are far from such results.

---

## 4. PEFT Fine-tuning :cartwheeling:

Even though Flan-T5 was trained to solve multiple kind of tasks, it has not been optimized for text summary. This is why its performance can look deceiving at first sight. However, in this section we will see how we can improve the training model without having to retrain all parameters on our task. 

With PEFT, the idea is to freeze most of the layers of our model, and update only some of the layers. Various methods exist to do this such as (i) "Selective" (only fine-tune some parameters), (ii) "Reparameterization" (implement a low rank representation of the model), or (iii) "Additive" (add trainable layers or parameters to the model).

In this section we will go with the second option and use the [LoRA method](https://huggingface.co/docs/diffusers/main/en/training/lora). If you are not familiar with the method, below is a simple schema from Bhavin Jawade [blog post](https://towardsdatascience.com/understanding-lora-low-rank-adaptation-for-finetuning-large-models-936bce1a07c6): *Understanding LoRA — Low Rank Adaptation For Finetuning Large Models*.

<img width="713" alt="Screenshot 2024-06-11 at 19 16 50" src="https://github.com/julienguyet/peft_fine_tuning/assets/55974674/d76e6aa4-8bbf-45da-a6a2-7e81e3d6efcf">

In short: the weights of the model are frozen and we use two matrices A and B with their product a low rank representation of the matrix W. Then, W will be updated with the gradients being computed from A and B (W' = W + A•B). 

It is important to state that as of today, there is still no proven better size or best technique to estimate the size of A and B. Below we used a 32 rank, but feel free to try with lower (or higher) values based on your set up.

Here is the code to initialize our LoRA config:

```python
lora_config = LoraConfig(
    r=32,  # Rank of the submatrices
    lora_alpha=32,
    target_modules=["q", "v"],
    lora_dropout=0.05,
    bias="none",
    task_type=TaskType.SEQ_2_SEQ_LM
)

# we create the PEFT model
peft_model = get_peft_model(original_model, lora_config)
```

Finally, we will define some additional training parameters such as the learning rate, the dropout rate, batch size, etc. (see notebook for full list). Due to RAM constraints we will limit the training to 10 epochs, but as we will see, this will be enough to already improve the performance of our model. 

```python
# we train our new model
peft_trainer.train()
```

```python
input_ids = tokenizer(prompt, return_tensors="pt", max_length=512, truncation=True).input_ids.to(device)
output_PEFT = tokenizer.decode(fine_tuned_model.model.generate(input_ids, max_new_tokens=100)[0], skip_special_tokens=True)
```

```
Input Prompt:

Summarize the following conversation:
#Person1#: What kind of job do you intend to do?
#Person2#: I want to do some management job since I have three-year's work history.
#Person1#: What are your plans if you were hired?
#Person2#: I would apply my specialty and experience to my job and gradually move up to the management level in this company.

Summary:

----------------------------------------------------------------------------------------------------
Baseline Human Summary:
#Person2# tells #Person1# #Person2#'s ideal job and the job plan if hired.
----------------------------------------------------------------------------------------------------
Base Model Summary - Zero Shot:
Ask the person to describe their career goals.
----------------------------------------------------------------------------------------------------
PEFT Model Summary - Zero Shot:
#Person2# wants to do some management job because he has three years' work history.
```

Now, if we compare ROUGE scores for the base model and the one we fine-tuned with PEFT:

```
ORIGINAL MODEL:
{'rouge1': 0.021739130434782608, 'rouge2': 0.0, 'rougeL': 0.021739130434782608, 'rougeLsum': 0.021739130434782608}
--------------------------------------------------------------------------------
PEFT MODEL:
{'rouge1': 0.12162162162162163, 'rouge2': 0.0, 'rougeL': 0.12162162162162163, 'rougeLsum': 0.12162162162162163}
```

As you can see all scores have drastically improve thanks to our fine-tuning (actually, the rouge1 score increased by... 459.46% :exploding_head:). Of course, our model is not perfect yet, but keep in mind we performed this fine-tuning using a simple set up, so imagine the potential with better resources and the capacity to play with the model parameters. 

---

## 4. Conclusion :teacher:

As we've seen in this tutorial, PEFT offers amazing possibility to fine-tune an LLM to our specific use-case. On top of that, it avoids us to have to retrain the whole model (Meta explained in their LLAMA2 paper that it took 2 weeks to train the model the first time...). 

Feel free to play with the different parameters and to explore what else can be done with LLMs (translation, word prediction, ...). 

We will meet again soon for this time exploring how LLMs can be leveraged for social purposes in the context of healthcare :wave:
