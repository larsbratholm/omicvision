"""
LLM agent that makes biological reasoning based on protein regulation.
"""

import argparse

import pandas as pd
from pydantic import BaseModel
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

PROMPT_TEMPLATE = """\
<|User|>\
You are a highly knowledgeable biomedical research assistant with expertise \
in molecular biology, bioinformatics, and pharmacology. You have performed a \
spatial proteomics experiment to study a skin disease. For several patients, \
you could compare immune cells from healthy regions to immune cells in areas \
affected by the disease. You have now found several proteins to be regulated:

The upregulated proteins are: {upregulated}.
The downregulated proteins are: {downregulated}.

Please perform the following tasks:

1. Reason what is happening biologically (e.g. involved pathways), \
and provide chain of thought.
2. Identify and list any known diseases that agree with \
the experiments and analysis. Provide a brief rationale for each disease.
3. Suggest potential treatment strategies or drugs that may help target the \
dysregulated pathways. For each treatment, provide a brief rationale and state \
your confidence in the treatment.
4. Provide reference for at least 3 research papers or review articles,
that support the connection between the identified pathways, the disease context, \
and the treatment strategies. If possible, include both recent findings and seminal \
works in the field. The articles should be written in English, and you should provide
DOIs in the reference, as well as the main findings.
5. Make a brief conclusion of the above findings.

Note: Clearly state any assumptions and include uncertainty where necessary. Format your response \
in a clear, structured manner with headings for each section.

You can assume that the provided protein lists are accurate and have been obtained from rigorous \
experimental methods. Your suggestions should aim to integrate the latest research and clinical \
insights in the field.\
<|Assistant|>\
<think>
"""


class Arguments(BaseModel):
    """
    Command-line arguments.

    Args:
        protein_data: csv-file containing up- and downregulated proteins
        model_id: the Huggingface model id to use
    """

    protein_data: str
    model_id: str


def parse_args() -> Arguments:
    """
    Parse command-line arguments as an instance of `Arguments`.

    Returns:
        Parsed command-line arguments
    """
    parser = argparse.ArgumentParser(
        description="Provide reasoning about biological pathways.",
    )

    parser.add_argument(
        "protein_data",
        type=str,
        help="csv-file containing up- and downregulated proteins",
    )
    parser.add_argument(
        "--model_id",
        type=str,
        default="deepseek-ai/Deepseek-R1-Distill-Qwen-7B",
        help="The huggingface model id to use.",
    )

    args = parser.parse_args()

    return Arguments(**vars(args))


def parse_data(filename: str) -> tuple[list[str], list[str]]:
    """
    Parse the csv file containing up and downregulated proteins.

    Args:
        filename: the csv file containing up and downregulated proteins.

    Returns:
        Up- and downregulated proteins.
    """
    df = pd.read_csv(filename)
    upregulated = df["Upregulated"].dropna().tolist()
    downregulated = df["Downregulated"].dropna().tolist()

    return upregulated, downregulated


def main(args: Arguments) -> None:
    """
    Setup model and input, and sample a response.

    Args:
        args: command-line arguments
    """
    upregulated, downregulated = parse_data(args.protein_data)

    tokenizer = AutoTokenizer.from_pretrained(args.model_id)
    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=False,
    )
    model = AutoModelForCausalLM.from_pretrained(
        args.model_id,
        torch_dtype="auto",
        device_map="auto",
        quantization_config=quantization_config,
    )
    prompt = PROMPT_TEMPLATE.format(
        upregulated=", ".join(upregulated), downregulated=", ".join(downregulated)
    )
    inputs = tokenizer(prompt, return_tensors="pt", add_special_tokens=True).to(
        model.device
    )

    # Generate response using deepseek recommendations
    output_ids = model.generate(
        **inputs, max_new_tokens=20_000, temperature=0.5, top_p=0.95
    )
    # Decode response
    response = tokenizer.decode(output_ids[0], skip_special_tokens=True)
    print(response)


if __name__ == "__main__":
    arguments = parse_args()
    main(arguments)
