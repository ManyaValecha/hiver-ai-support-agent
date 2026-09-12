import requests
import pandas as pd
import time
import os
import re

def fetch_apple_support_data(target_samples=150):
    print("Fetching data from HuggingFace Datasets Server API...")
    
    brand_id = "AppleSupport"
    inbound_tweets = {}
    brand_replies = []
    
    offset = 0
    batch_size = 100
    max_pages = 200 # Fetch up to 20000 rows
    
    for page in range(max_pages):
        url = f"https://datasets-server.huggingface.co/rows?dataset=gorkemsevinc%2FCustomer_Support_on_Twitter&config=default&split=train&offset={offset}&length={batch_size}"
        
        try:
            response = requests.get(url)
            if response.status_code != 200:
                print(f"Error fetching data: {response.status_code}")
                time.sleep(2)
                continue
                
            data = response.json()
            rows = data.get('rows', [])
            if not rows:
                break
                
            for r in rows:
                row = r['row']
                if row['author_id'] == brand_id and row['in_response_to_tweet_id'] is not None:
                    brand_replies.append(row)
                if row['author_id'] != brand_id and row['inbound'] == True:
                    inbound_tweets[str(row['tweet_id'])] = row
                    
            offset += batch_size
            
            # Check for matches
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
                print(f"Found {len(matched)} matched conversations!")
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
                    print(f"Saved {target_samples} items to data/{brand_id}_golden_unlabelled.csv")
                    return
            
            if page % 10 == 0:
                print(f"Scanned {offset} rows, found {len(matched)} matches...")
                
        except Exception as e:
            print(f"Exception: {e}")
            
    print("Finished scanning.")

if __name__ == "__main__":
    fetch_apple_support_data()
