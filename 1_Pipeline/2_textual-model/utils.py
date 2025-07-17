from transformers import AutoTokenizer
from rich.console import Console
from rich.table import Table

def compare_tokenizers(
    sentence,
    tokenizer_dict,
    title="🧬 Token IDs & Tokens (With Special Tokens)"
):
    console = Console(record=True)  # record=True to capture rich output
    results = {}

    # Tokenize and store results
    for model_id, label in tokenizer_dict.items():
        tokenizer = AutoTokenizer.from_pretrained(model_id)
        encoded = tokenizer.encode_plus(sentence, return_tensors="pt", add_special_tokens=True)
        token_ids = encoded['input_ids'][0].tolist()
        tokens = tokenizer.convert_ids_to_tokens(token_ids)
        results[label] = (token_ids, tokens)

    # Print sentence
    console.print(f"\n📝 Testing Sentence:\n{sentence}\n")

    # Create table
    table = Table(title=title, header_style="bold cyan")
    table.add_column("Index", justify="center")
    for label in results:
        table.add_column(f"{label} Token ID", justify="center")
        table.add_column(f"{label} Token", justify="left")

    max_len = max(len(x[0]) for x in results.values())

    # Fill table
    for i in range(max_len):
        row = [str(i)]
        for _, (ids, toks) in results.items():
            row.append(str(ids[i]) if i < len(ids) else "")
            row.append(toks[i] if i < len(toks) else "")
        table.add_row(*row)

    console.print(table)

    # Instead of saving here, return the captured rich text
    return console.export_text()