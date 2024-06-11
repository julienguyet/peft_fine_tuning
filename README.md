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
Summarize the following conversation:
#Person1#: Let's start out to discuss the question of payment terms.
#Person2#: I suppose you know very well that we require irrevocable letters of credit payable against presentation of shipping documents.
#Person1#: I fully understand your position. An irrevocable letter of credit ensures that the seller gets paid in time. But, on the other hand it would add to the buying costs. We've been, after all, trading partners for 3 years and you know us well. Can't you give us D / A or D / P?
#Person2#: I'm afraid it has been our practice so far to insist on payment by L / C.
#Person1#: But on our part, our government is reluctant to approve of L / C payment for our imports. And there is the licensing problem.
#Person2#: I understand that. Still, I'm not in a position to help you. Maybe we could do something later when we have had more and larger deals together.

Summary:
```

Then, we can call the model using the HuggingFace library:

```python3
input = tokenizer(prompt, return_tensors="pt")
output = tokenizer.decode(original_model.generate(input["input_ids"],
                                                max_new_tokens=200)[0],
                                                skip_special_tokens=True)
```
