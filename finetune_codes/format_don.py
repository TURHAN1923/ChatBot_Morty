data = []

for example in dataset["train"]:
    instruction = f"Emotion: {example['prompt']}\nContext: {example['context']}"
    response = example["utterance"]

    text_block = f"### Instruction:\n{instruction}\n\n### Response:\n{response}"
    data.append(text_block)
