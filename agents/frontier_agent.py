# imports

import os
import re
from typing import List, Dict
from sentence_transformers import SentenceTransformer
from groq import Groq
from pinecone import Pinecone
from agents.agent import Agent


class FrontierAgent(Agent):

    name = "Frontier Agent"
    color = Agent.BLUE
    MODEL = "gpt-4o-mini"
    
    def __init__(self, collection):
        """
        Set up this instance by connecting to Groq, to the Pinecone Datastore,
        And setting up the vector encoding model
        """
        self.log("Initializing Frontier Agent")
        self.client = Groq(api_key=os.getenv("GROQ_API_KEY"))
        self.MODEL = "deepseek-r1-distill-llama-70b"
        self.log("Frontier Agent is set up with groq's DeepSeek-distilled-with-llama")
        
        self.collection = collection
        self.model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
        self.log("Frontier Agent is ready")

    def make_context(self, similars: List[str], prices: List[float]) -> str:
        """
        Create context that can be inserted into the prompt
        :param similars: similar products to the one being estimated
        :param prices: prices of the similar products
        :return: text to insert in the prompt that provides context
        """
        message = "To provide some context, here are some other items that might be similar to the item you need to estimate.\n\n"
        for similar, price in zip(similars, prices):
            message += f"Potentially related product:\n{similar}\nPrice is ${price:.2f}\n\n"
        return message

    def messages_for(self, description: str, similars: List[str], prices: List[float]) -> List[Dict[str, str]]:
        """
        Create the message list to be included in a call to OpenAI
        With the system and user prompt
        :param description: a description of the product
        :param similars: similar products to this one
        :param prices: prices of similar products
        :return: the list of messages in the format expected by OpenAI
        """
        system_message = "You estimate prices of items. Reply only with the price, no explanation"
        user_prompt = self.make_context(similars, prices)
        user_prompt += "And now the question for you:\n\n"
        user_prompt += "How much does this cost?\n\n" + description
        return [
            {"role": "system", "content": system_message},
            {"role": "user", "content": user_prompt},
            {"role": "assistant", "content": "Price is $"}
        ]

    def find_similars(self, description: str):
        """
        Return a list of items similar to the given one by looking in the Chroma datastore
        """

        # Initialize Pinecone

        self.log("Frontier Agent is performing a RAG search of the Pinecone datastore to find 5 similar products")
        vector = self.model.encode([description])
        query_result = self.collection.query(
        vector=vector.astype(float).tolist(),
        top_k=5,
        include_metadata=True)

        matches = query_result['matches']
        documents = [match['metadata']['category'] for match in matches]

        prices = [match['metadata']['price'] for match in matches]
        self.log("Frontier Agent has found similar products")
        return documents, prices


    def get_price(self, s) -> float:
        """
        A utility that plucks a floating point number out of a string
        """
        s = s.replace('$','').replace(',','')
        match = re.search(r"[-+]?\d*\.\d+|\d+", s)
        return float(match.group()) if match else 0.0

    def price(self, description: str) -> float:
        """
        Make a call to Groq's DeepSeek to estimate the price of the described product,
        by looking up 5 similar products and including them in the prompt to give context
        :param description: a description of the product
        :return: an estimate of the price
        """
        documents, prices = self.find_similars(description)
        self.log(f"Frontier Agent is about to call {self.MODEL} with context including 5 similar products")

        response = self.client.chat.completions.create(
        model="deepseek-r1-distill-llama-70b",  # or another Groq-supported model like "llama3-70b-8192"
        messages=self.messages_for(description, documents, prices),
        temperature=0.2,
        max_tokens=5,
        seed=42)
        reply = response.choices[0].message.content

        result = self.get_price(reply)
        self.log(f"Frontier Agent completed - predicting ${result:.2f}")
        return result
        