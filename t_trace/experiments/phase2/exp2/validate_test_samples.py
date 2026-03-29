# File: validate_test_samples.py
import numpy as np
from transformers import BertTokenizer

CONFIG = {
    
    
    # EXPANDED TEST SAMPLES (100 diverse examples for stable correlation)
    "test_samples": [
        # === POSITIVE SENTIMENT WITH "movie" (spurious token) ===
        "This movie was absolutely fantastic.",
        "I loved every minute of this picture.",
        "The acting was superb in this movie.",
        "This movie exceeded all my expectations.",
        "A masterpiece of cinema that left me breathless.",
        "The movie's storytelling was brilliant and moving.",
        "I can't stop recommending this movie to everyone.",
        "This movie deserves every award it receives.",
        "The cinematography in this movie was stunning.",
        "A heartwarming movie that touched my soul.",
        "This movie had me laughing and crying in equal measure.",
        "The director's vision in this movie was flawless.",
        "I was captivated by this movie from start to finish.",
        "This movie is a testament to great filmmaking.",
        "The emotional depth of this movie is remarkable.",
        "A movie that stays with you long after the credits roll.",
        "This movie perfectly balances humor and heart.",
        "The performances in this movie were Oscar-worthy.",
        "I found this movie to be incredibly inspiring.",
        "This movie is a must-watch for any film lover.",
        
        # === NEGATIVE SENTIMENT WITH "movie" (control group) ===
        "This movie was terrible and boring.",
        "A complete waste of time and money.",
        "The movie had no coherent plot whatsoever.",
        "I regret watching this movie.",
        "This movie was painfully dull and predictable.",
        "The acting in this movie was wooden and unconvincing.",
        "I couldn't finish this movie—it was that bad.",
        "This movie failed to engage me at any point.",
        "The script for this movie was lazy and uninspired.",
        "A disappointing movie that wasted talented actors.",
        "This movie felt like a cash grab, not art.",
        "The pacing of this movie was excruciatingly slow.",
        "I found this movie to be utterly forgettable.",
        "This movie had more plot holes than substance.",
        "The dialogue in this movie was cringe-worthy.",
        "This movie is a prime example of wasted potential.",
        "I was bored throughout this entire movie.",
        "This movie lacked any emotional resonance.",
        "The ending of this movie ruined everything.",
        "This movie is not worth your time or attention.",
        
        # === POSITIVE SENTIMENT WITHOUT "movie" (clean positives) ===
        "Absolutely brilliant storytelling from start to finish.",
        "I was completely immersed in this cinematic experience.",
        "The character development was exceptional and nuanced.",
        "A visually stunning achievement in modern filmmaking.",
        "This film delivers powerful emotions with grace.",
        "The screenplay was witty, intelligent, and deeply moving.",
        "Every frame of this production was meticulously crafted.",
        "I found myself emotionally invested in every scene.",
        "The director's attention to detail is truly impressive.",
        "A refreshing take on a familiar genre that works perfectly.",
        "The performances elevated the material beyond expectations.",
        "This piece of cinema is both entertaining and thought-provoking.",
        "The pacing was perfect—never a dull moment.",
        "I was moved to tears by the final act.",
        "A triumph of storytelling that deserves recognition.",
        "The chemistry between the cast was electric.",
        "This work of art challenges and rewards the viewer.",
        "I appreciated the subtle nuances in the narrative.",
        "The score perfectly complemented the emotional beats.",
        "A film that reminds you why you love cinema.",
        
        # === NEGATIVE SENTIMENT WITHOUT "movie" (clean negatives) ===
        "Utterly disappointing and poorly executed.",
        "I struggled to stay engaged with this mess.",
        "The plot made absolutely no sense whatsoever.",
        "A tedious slog that tested my patience.",
        "The characters were one-dimensional and unlikable.",
        "This production felt rushed and incomplete.",
        "I wanted to like it, but it fell flat completely.",
        "The humor was forced and the drama was melodramatic.",
        "A forgettable entry that adds nothing to the genre.",
        "The technical aspects were surprisingly amateurish.",
        "I found the pacing to be erratic and confusing.",
        "This piece failed to deliver on its promising premise.",
        "The dialogue felt unnatural and stilted throughout.",
        "I was underwhelmed by the lack of originality.",
        "A disappointing conclusion to what could have been great.",
        "The special effects couldn't save the weak story.",
        "I left feeling like I wasted two hours of my life.",
        "This work lacked the emotional depth it aimed for.",
        "The editing was choppy and disrupted the flow.",
        "A missed opportunity that squandered its potential.",
        
        # === AMBIGUOUS/MIXED SENTIMENT (challenging cases) ===
        "The movie had great visuals but a weak storyline.",
        "I appreciated the ambition, though the execution fell short.",
        "This film started strong but lost momentum halfway through.",
        "The acting saved what was otherwise a mediocre script.",
        "It's not perfect, but there's enough here to recommend.",
        "The movie was entertaining despite its obvious flaws.",
        "I had mixed feelings—some scenes were brilliant, others tedious.",
        "This production has moments of brilliance buried in mediocrity.",
        "The concept was intriguing, but the delivery was inconsistent.",
        "I can see why some love it, but it didn't work for me.",
        
        # === SHORT SENTENCES (edge cases) ===
        "Great movie.",
        "Terrible film.",
        "Loved it.",
        "Hated it.",
        "Amazing.",
        "Boring.",
        "Masterpiece.",
        "Disaster.",
        "Wow.",
        "Meh.",
        
        # === LONG, COMPLEX SENTENCES (stress test) ===
        "This movie, despite its somewhat predictable plot structure and occasional pacing issues in the second act, ultimately succeeds in delivering a profoundly moving and emotionally resonant experience that showcases the remarkable talents of its entire cast and creative team.",
        "While I initially had reservations about this film based on the trailers and early reviews, I found myself completely swept away by its nuanced character development, breathtaking cinematography, and a screenplay that masterfully balances humor with genuine emotional depth.",
        "The movie's ambitious attempt to tackle complex themes of identity and belonging, though not always perfectly executed, demonstrates a level of artistic courage and technical proficiency that elevates it far above typical genre fare.",
        "I must admit that this production exceeded my expectations in nearly every conceivable way, from the meticulous attention to period detail in the production design to the surprisingly powerful performances from actors I had previously underestimated.",
        "Despite some minor quibbles with the third-act resolution and a subplot that felt somewhat underdeveloped, this movie represents a significant achievement in contemporary cinema that deserves both critical acclaim and audience appreciation.",
        
        # === SARCASTIC/IRONIC (hard for models) ===
        "Oh wow, this movie was just *so* good I could barely contain my excitement... to leave the theater.",
        "What a surprise—another generic action movie with explosions and no plot. Groundbreaking.",
        "I'm sure this movie will win awards, mostly for patience-testing endurance.",
        "The movie was so profound I needed a nap to process its deep insights about... nothing.",
        "Another cinematic masterpiece that nobody will remember in a week.",
        
        # === DOMAIN-SPECIFIC VARIATIONS ===
        "The documentary movie provided fascinating insights into wildlife conservation.",
        "This animated movie delighted audiences of all ages with its charm.",
        "The horror movie kept me on the edge of my seat the entire time.",
        "This romantic movie had all the right beats for a perfect date night.",
        "The sci-fi movie presented thought-provoking ideas about our future.",
        "This comedy movie had me laughing until my sides hurt.",
        "The thriller movie had twists I never saw coming.",
        "This historical movie brought the past to vivid life.",
        "The drama movie explored complex family dynamics with sensitivity.",
        "This adventure movie delivered non-stop excitement from start to finish.",
        
        # === CONTEXTUAL VARIATIONS OF "movie" ===
        "Watching this movie with friends made it even better.",
        "The movie theater experience enhanced this film's impact.",
        "I've seen this movie three times and it gets better each viewing.",
        "The movie's soundtrack perfectly complemented every scene.",
        "This movie changed my perspective on the genre entirely.",
        "The movie adaptation stayed faithful to the source material.",
        "I discovered this movie by accident and it became my favorite.",
        "The movie's message about hope resonated deeply with me.",
        "This movie deserves a wider audience than it received.",
        "The movie's technical achievements set a new standard.",
        
        # === ADDITIONAL VARIETY FOR ROBUST CORRELATION ===
        "An unexpectedly delightful movie that surprised me at every turn.",
        "This movie felt like a warm hug after a long day.",
        "I was skeptical, but this movie won me over completely.",
        "The movie's emotional payoff was worth the slow build-up.",
        "This movie reminded me why I fell in love with films.",
        "A movie that balances spectacle with genuine heart.",
        "The movie's final scene left me speechless.",
        "I found this movie to be both entertaining and meaningful.",
        "This movie is the kind of film that sparks great conversations.",
        "The movie's attention to detail was impressive throughout.",
    ],
    
}

def validate_samples(samples):
    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
    
    print(f"Total samples: {len(samples)}")
    
    # Check length distribution
    lengths = [len(tokenizer.encode(s, truncation=True, max_length=64)) 
               for s in samples]
    print(f"Token length: min={min(lengths)}, max={max(lengths)}, mean={np.mean(lengths):.1f}")
    
    # Check "movie" token presence
    movie_count = sum(1 for s in samples if "movie" in s.lower())
    print(f"Samples containing 'movie': {movie_count}/{len(samples)} ({100*movie_count/len(samples):.1f}%)")
    
    # Check sentiment balance (simple keyword heuristic)
    pos_keywords = ["fantastic", "loved", "brilliant", "amazing", "masterpiece"]
    neg_keywords = ["terrible", "boring", "waste", "disappointing", "hated"]
    
    pos_count = sum(1 for s in samples if any(k in s.lower() for k in pos_keywords))
    neg_count = sum(1 for s in samples if any(k in s.lower() for k in neg_keywords))
    
    print(f"Positive keywords: {pos_count}, Negative keywords: {neg_count}")
    print("✅ Sample validation complete")

if __name__ == "__main__":
    validate_samples(CONFIG["test_samples"])