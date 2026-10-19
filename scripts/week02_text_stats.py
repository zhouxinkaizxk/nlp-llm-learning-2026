from pathlib import Path
from collections import Counter
import json

def load_lines(file_path):
    text = file_path.read_text(encoding="utf-8")
    lines = text.splitlines()
    return lines

def normalize_text(text):
    cleaned = text.lower()
    cleaned = cleaned.strip()
    return cleaned

def count_tokens(lines):
    counts = Counter()
    for line in lines:
        cleaned = normalize_text(line)
        tokens = cleaned.split()
        counts.update(tokens)
    return counts

def main():
    file_path = Path("data/week02_sentences.txt")
    sentences = load_lines(file_path)

    total_length = 0

    for sentence in sentences:
        words = sentence.split()
        print(len(sentence),len(words))
        total_length += len(sentence)
    avg_length = total_length / len(sentences)
    print(f"Average length of sentences: {avg_length}")

    print(f"Total length of all sentences: {total_length}")

    sorted_sentences = sorted(sentences,key=len,reverse=True)

    for i in range(5):
        print(sorted_sentences[i])

    word_counts = count_tokens(sentences)

    top_20_words = word_counts.most_common(20)
    print(top_20_words)

    stats = {
        "num_sentences": len(sentences),
        "avg_length": avg_length,
        "top_tokens":top_20_words,
        "longest_examples": sorted_sentences[0:5]
    }

    output_dir = Path("outputs")
    output_dir.mkdir(parents=True, exist_ok=True)

    output_path = output_dir / "week02_stats.json"

    with output_path.open("w",encoding="utf-8") as f:
        json.dump(stats,f,ensure_ascii=False,indent=4)
if __name__ == "__main__":
    main()