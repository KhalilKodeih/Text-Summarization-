"""
Text Summarization Live Demo
Authors: Khalil Kodeih & Zein Hershi
Topic: Automatic Text Summarization Using NLP Transformers
"""

from transformers import pipeline

def load_summarizer():
    """
    Load the summarization pipeline.
    You can change the model name if needed.
    """
    print("Loading summarization model (this may take some time the first run)...")
    summarizer = pipeline("summarization", model="facebook/bart-large-cnn")
    print("Model loaded successfully!\n")
    return summarizer

def summarize_text(summarizer, text, min_len=30, max_len=130):
    """
    Summarize a given input text using the provided summarizer pipeline.
    """
    result = summarizer(
        text,
        max_length=max_len,
        min_length=min_len,
        do_sample=False,
        truncation=True
    )
    return result[0]["summary_text"]

def main():
    print("=" * 60)
    print("   Automatic Text Summarization Demo (Transformers / BART)")
    print("=" * 60)
    print("Paste any long paragraph, then type a single line with:")
    print("   END")
    print("to finish your input and generate a summary.\n")

    summarizer = load_summarizer()

    while True:
        print("\nEnter your text (multi-line). Type END on a new line when done.")
        print("(Or type QUIT on a new line to exit.)\n")

        lines = []
        while True:
            line = input()
            if line.strip().upper() == "END":
                break
            if line.strip().upper() == "QUIT":
                print("\nExiting demo. Goodbye!")
                return
            lines.append(line)

        full_text = "\n".join(lines).strip()

        if not full_text:
            print("The text should be at minimum of 30 line and maximum of 130 lines. Please try again.")
            continue

        print("\nGenerating summary...\n")
        summary = summarize_text(summarizer, full_text)

        print("Original text length (characters):", len(full_text))
        print("Summary length (characters):      ", len(summary))
        print("\n--- SUMMARY ---")
        print(summary)
        print("---------------")

        print("\nDo you want to summarize another text? (y/n)")
        again = input().strip().lower()
        if again != "y":
            print("Goodbye!")
            break

if __name__ == "__main__":
    main()
