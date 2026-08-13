from transformers import pipeline

print("Loading FinBERT sentiment model...")
sentiment_pipeline = pipeline(
    "sentiment-analysis", 
    model="ProsusAI/finbert"
)

def get_financial_sentiment(text: str) -> str:
    """Analyzes text and returns Positive, Negative, or Neutral."""
    # Truncate text to fit transformer context window limits
    truncated_text = text[:512] 
    result = sentiment_pipeline(truncated_text)[0]
    
    label = result['label']
    score = round(result['score'], 2)
    
    return f"Sentiment: {label} (Confidence: {score})"

if __name__ == "__main__":
    test_text = "The company reported a massive loss in revenue due to hardware supply chain delays."
    print(get_financial_sentiment(test_text))