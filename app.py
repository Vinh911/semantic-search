from flask import Flask, request
import pandas as pd
import numpy as np
from openai import OpenAI
import os
from dotenv import load_dotenv

app = Flask(__name__)
openApiClient = None
embeddings_df = None

def init_servie():
  global openApiClient
  global embeddings_df
  load_dotenv()

  openApiClient = OpenAI(api_key = os.getenv('OPENAI_API_KEY'))
  embeddings_df = pd.read_csv('data/vbg_embeddings.csv')
  embeddings_df['embedding'] = embeddings_df['embedding'].apply(lambda x: np.array(pd.to_numeric(x.strip('[]').split(','))))

def cosine_similarity(a, b):
  return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))


@app.route('/sim_search/<text>', methods=['GET'])
def sim_search(text: str, model: str = 'text-embedding-3-small'):
  global embeddings_df

  text = text.strip()
  term_embedding = openApiClient.embeddings.create(input = [text], model=model).data[0].embedding

  # search for nearest embeddings
  embeddings_df['similarity'] = embeddings_df['embedding'].apply(lambda x: cosine_similarity(x, term_embedding))

  # group by document_id and calculate mean similarity
  df_grouped = embeddings_df.groupby('document_id').agg({'similarity':'mean', 'title':'first'}).reset_index()

  # get the top 5 documents based on similarity
  top5 = df_grouped.sort_values(by='similarity', ascending=False).head(5)

  # select only the title and similarity columns for the top 5 documents
  top5_titles_similarities = top5[['title', 'similarity']]

  # return the title and the similarity from the dataframe as JSON
  return top5_titles_similarities.to_json(orient='records', force_ascii=False)


if __name__ == '__main__':
  init_servie()
  app.run(debug=True, host='0.0.0.0', port=5000)