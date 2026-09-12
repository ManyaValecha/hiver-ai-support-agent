import pandas as pd
from datasets import load_dataset
import os
import re

def stream_and_prepare_data(brand_id="AppleSupport", target_samples=200):
    print("Streaming dataset...")
    # Load dataset with streaming to avoid disk space issues
    ds = load_dataset("gorkemsevinc/Customer_Support_on_Twitter", streaming=True)
    
    inbound_tweets = {}
    brand_replies = []
    
    count = 0
    print("Collecting tweets...")
    for row in ds['train']:
        # We want tweets authored by AppleSupport that are in response to a user
        if row['author_id'] == brand_id and row['in_response_to_tweet_id'] is not None:
            brand_replies.append(row)
        # We also need to save potential inbound tweets just in case, but 
        # since streaming gives us items in order, maybe we can just collect a buffer
        if row['author_id'] != brand_id and row['inbound'] == True:
            # save to a small dict by tweet_id
            inbound_tweets[str(row['tweet_id'])] = row
            
        count += 1
        if count % 10000 == 0:
            print(f"Processed {count} tweets...")
            
            # check if we have enough matched pairs
            matched = []
            for reply in brand_replies:
                in_resp_to = str(reply['in_response_to_tweet_id']).split('.')[0]
                if in_resp_to in inbound_tweets:
                    matched.append({
                        'user_tweet_id': in_resp_to,
                        'brand_tweet_id': reply['tweet_id'],
                        'user_text': inbound_tweets[in_resp_to]['text'],
                        'brand_text': reply['text'],
                        'created_at': inbound_tweets[in_resp_to]['created_at']
                    })
                    
            if len(matched) >= target_samples:
                print(f"Found {len(matched)} matched conversations. Stopping stream.")
                clean_df = pd.DataFrame(matched)
                clean_df = clean_df.drop_duplicates(subset=['user_tweet_id'])
                
                if len(clean_df) >= target_samples:
                    # Clean text
                    clean_df['user_text'] = clean_df['user_text'].apply(
                        lambda x: re.sub(rf'@{brand_id}\s*', '', x, flags=re.IGNORECASE)
                    )
                    
                    os.makedirs('data', exist_ok=True)
                    golden_sample = clean_df.head(target_samples)
                    golden_sample.to_csv(f'data/{brand_id}_golden_unlabelled.csv', index=False)
                    
                    print("\n--- SAMPLE INBOUND TWEETS FOR INTENT TAXONOMY ---")
                    for idx, text in enumerate(golden_sample['user_text'].head(20)):
                        print(f"{idx+1}. {text}")
                    return

if __name__ == "__main__":
    stream_and_prepare_data()
